        raise SystemExit("Dataset is empty.")

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Loading model...")
    load_args = {"torch_dtype": "auto"}
    if args.device == "auto":
        load_args["device_map"] = "auto"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
        load_args["device_map"] = {"": 0}
    else:
        load_args["device_map"] = {"": "cpu"}
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_args)
    model.eval()
    model_device = next(model.parameters()).device
    model_revision = getattr(model.config, "_commit_hash", None) or "unknown"
    print(f"Model ready on {model_device}; revision={model_revision}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id", "fact_group_id", "language", "fact_type", "method",
        "model", "model_revision", "prompt_tokens", "cache_seq_len",
        "cache_bytes_estimate", "expected_answer", "response", "exact_match",
        "filler_repeats", "generation_seconds", "error"
    ]
    write_header = not out_path.exists() or out_path.stat().st_size == 0

    with out_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()

        for idx, item in enumerate(records, 1):
            row = {key: "" for key in fields}
            row.update({
                "case_id": item.get("case_id", ""),
                "fact_group_id": item.get("fact_group_id", ""),
                "language": item.get("language", ""),
                "fact_type": item.get("fact_type", ""),
                "method": "full_kv",
                "model": args.model,
                "model_revision": model_revision,
                "expected_answer": item["answer"],
                "filler_repeats": args.filler_repeats,
                "error": "",
            })
            try:
                context, question = make_prompt(item, args.filler_repeats)
                # Chat template matches instruction-tuned model expectations.
                messages = [
                    {"role": "system", "content": "Answer the question using only the provided context. Return the requested value exactly and nothing else. If unsure, say UNKNOWN."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt")
                prompt_tokens = int(inputs["input_ids"].shape[-1])
                inputs = {k: v.to(model_device) for k, v in inputs.items()}

                start = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        return_dict_in_generate=True,
                    )
                elapsed = time.perf_counter() - start
                generated = output.sequences[0, prompt_tokens:]
                response = tokenizer.decode(generated, skip_special_tokens=True).strip()

                cache_len, cache_bytes = get_cache_stats(
                    getattr(output, "past_key_values", None)
                )
                row.update({
                    "prompt_tokens": prompt_tokens,
                    "cache_seq_len": cache_len if cache_len is not None else "",
                    "cache_bytes_estimate": cache_bytes if cache_bytes is not None else "",
                    "response": response,
                    "exact_match": int(answer_match(item["answer"], response)),
                    "generation_seconds": round(elapsed, 4),
                })
                print(f"[{idx}/{len(records)}] {item['case_id']} {item['language']} "
                      f"match={row['exact_match']} response={response[:100]!r}")
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                print(f"[{idx}/{len(records)}] ERROR {item.get('case_id')}: {row['error']}")
            writer.writerow(row)
            f.flush()

    print(f"\nWrote results to: {out_path}")
    print("Reminder: inspect generated responses; automatic exact-match is only a first-pass metric.")

if __name__ == "__main__":
    main()
