from utils.data_tool import get_topic_score_by_year
from utils import paths

if __name__ == "__main__":
    # Calculate scores for 20 topics by year
    get_topic_score_by_year(paths.lda_model_path, paths.id2word_path,
                            source_dir=paths.train_data_path,
                            target_dir=paths.train_output_dir)
    get_topic_score_by_year(paths.lda_model_path, paths.id2word_path,
                            source_dir=paths.test_data_path,
                            target_dir=paths.test_output_dir)
