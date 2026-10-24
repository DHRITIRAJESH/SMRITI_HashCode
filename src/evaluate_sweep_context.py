        language = get_field(row, ["language", "lang"], required=False)
        fact_type = get_field(row, ["fact_type", "type"], required=False)
        fact_text = get_field(row, ["fact", "text", "statement"])
        question = get_field(row, ["question"])
        expected = get_field(row, ["expected", "answer", "target"])

        for requested_len in lengths:
            key = (case_id, requested_len)
            if key in completed:
                continue

            filler = filler_by_length[requested_len]
            input_ids, messages = render_messages(tokenizer, fact_text, filler, question)
            prompt_tokens = int(input_ids.shape[-1])

            # Inputs are moved to the model's input embedding device.
            input_device = model.get_input_embeddings().weight.device
            input_ids = input_ids.to(input_device)
            attention_mask = torch.ones_like(input_ids)

            t0 = time.time()
            with torch.inference_mode():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,  # full-cache baseline
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = output[0, input_ids.shape[-1]:]
            prediction = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            elapsed = time.time() - t0

            pred_norm = normalize_basic(prediction)
            exp_norm = normalize_basic(expected)
            # Strict baseline flag: whitespace-normalized string equality only.
            strict_match = pred_norm == exp_norm
            # Recall/recoverability signal only, not exact correctness. Conservative
            # substring check is retained as a diagnostic, not a primary metric.
            recoverable = bool(exp_norm and exp_norm in pred_norm)

            result = {
                "case_id": case_id,
                "fact_group_id": fact_group_id,
                "language": language,
                "fact_type": fact_type,
                "requested_filler_tokens": requested_len,
                "measured_prompt_tokens": prompt_tokens,
                "fact": fact_text,
                "question": question,
                "expected": expected,
                "predicted": prediction,
                "strict_match_whitespace_normalized": strict_match,
                "expected_value_appears": recoverable,
                "elapsed_seconds": round(elapsed, 3),
                "cache_mode": "full",
                "target_position": "beginning_of_user_record",
                "seed": args.seed,
                "model": args.model,
            }
            results.append(result)
            completed.add(key)
            done += 1

            # Incremental checkpoint after every evaluation.
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(out_path)

            print(
                f"[{done}/{total_jobs}] {case_id} filler={requested_len} "
                f"prompt_tokens={prompt_tokens} strict={strict_match} "
                f"recoverable={recoverable} pred={prediction[:100]!r}",
                flush=True,
            )

    print("\n" + "=" * 68)
    print("CONTEXT SWEEP SUMMARY")
    print("=" * 68)
    for requested_len in lengths:
        subset = [r for r in results if r["requested_filler_tokens"] == requested_len]
        strict = sum(r["strict_match_whitespace_normalized"] is True for r in subset)
        recov = sum(r["expected_value_appears"] is True for r in subset)
        n = len(subset)
        print(
            f"filler={requested_len:5d}  n={n:3d}  "
            f"strict={strict}/{n} ({strict/n*100:.2f}%)  "
            f"recoverable={recov}/{n} ({recov/n*100:.2f}%)"
            if n else f"filler={requested_len}: no results"
        )

    print(f"\nSaved: {out_path}", flush=True)
    print("Note: exact equality here is whitespace-normalized only. "
          "Audit answer normalization before making research claims.")


if __name__ == "__main__":
    main()
