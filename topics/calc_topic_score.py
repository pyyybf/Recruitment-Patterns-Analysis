from utils.data_tool import get_topic_score
from utils import paths

if __name__ == "__main__":
    # Calculate scores for 20 topics by file
    get_topic_score(paths.lda_model_path, paths.id2word_path,
                    source_dir=paths.train_data_path,
                    change_rate_file_path=f"{paths.train_data_path}/train_change_rate_unfilled.csv",
                    target_dir=paths.train_output_dir,
                    prefix="train")
    get_topic_score(paths.lda_model_path, paths.id2word_path,
                    source_dir=paths.test_data_path,
                    change_rate_file_path=f"{paths.test_data_path}/test_change_rate_unfilled.csv",
                    target_dir=paths.test_output_dir,
                    prefix="test")
