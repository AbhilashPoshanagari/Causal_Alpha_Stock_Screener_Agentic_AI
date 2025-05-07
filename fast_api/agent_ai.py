from team_2_stock_recommendation_agentic_v6_all_agent_logged import gradio_predict

def predict_llm(stock_ticker, analysis_type, n_runs, check_similarity, similarity_threshold,
                            compare_fields, use_cache, show_similarity_summary, add_weights,
                            macro_weight_val, sector_weight_val, tech_weight_val):
    predict_re = gradio_predict(stock_ticker=stock_ticker, analysis_type=analysis_type, n_runs=n_runs, check_similarity=check_similarity, similarity_threshold=similarity_threshold,
                            compare_fields=compare_fields, use_cache=use_cache, show_similarity_summary=show_similarity_summary, add_weights=add_weights,
                            macro_weight_val=macro_weight_val, sector_weight_val=sector_weight_val, tech_weight_val=tech_weight_val )
    # print(predict_re)
    return predict_re