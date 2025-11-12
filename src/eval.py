        agent_output_tree = parse_agent_outputs(run_config.run_dir_path / "outputs")

        client = create_azure_model(framework="openai")

        logging.info(f"Running judge LLM to evaluate the results")
        if run_config.task_id == "giab":
            agent_results = eval_giab_metrics(
                run_config.run_dir_path / "results",
                run_config.data_path / "results",
                inputs_root / "data" / "Agilent_v7.chr.bed",
                inputs_root / "reference" / "Homo_sapiens_assembly38.fasta",
            )
            judge_prompt = build_judge_prompt_giab(
                input_data,
                run_config.task_prompt,
                agent_output_tree,
                agent_results,
            )
            completion = client.beta.chat.completions.parse(
                model="gpt-5",
                messages=[{"role": "user", "content": judge_prompt}],
                response_format=EvaluationResultsGiabSchema,
            )
            parsed_response = completion.choices[0].message.parsed
            final_result = EvaluationResultsGiab(
                steps_completed=parsed_response.steps_completed,
                steps_to_completion=parsed_response.steps_to_completion,
                final_results_reached=parsed_response.final_results_reached,
                f1_score=parsed_response.f1_score,
                notes=parsed_response.notes,
            )

        else:
            agent_results = parse_agent_results(run_config.run_dir_path / "results")
            truth_results = parse_agent_results(run_config.data_path / "results")
            judge_prompt = build_judge_prompt_csv(
                input_data,
                run_config.task_prompt,
                agent_output_tree,
                agent_results,
                truth_results,
            )
            completion = client.beta.chat.completions.parse(
                model="gpt-5",
                messages=[{"role": "user", "content": judge_prompt}],
                response_format=EvaluationResultsSchema,
            )
            parsed_response = completion.choices[0].message.parsed
            final_result = EvaluationResults(
                steps_completed=parsed_response.steps_completed,
                steps_to_completion=parsed_response.steps_to_completion,
                final_result_reached=parsed_response.final_result_reached,
                notes=parsed_response.notes,
            )
        logging.info(f"Judge LLM finished running with results: {final_result}")
        run_config.eval_results = final_result
        run_config.save_run_metadata()
        logging.info(f"Run configuration saved with results: {run_config.eval_results}")