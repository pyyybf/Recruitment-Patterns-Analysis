from utils import processor

def main():
    # 测试的路径，可以改成自己的路径，这里是我本地的路径
    test_filepath = "/Users/weichentao/Documents/USC/2023fall/540/project/data_txt/2016/1750_000104746916014299_a2228768z10-k.htm.txt"
    
    processed_bag = processor.process_file(test_filepath)
    
    processed_list = processed_bag.compute()
    
    for i, line in enumerate(processed_list):
        if i >= 100:
            break
        print(line)

if __name__ == '__main__':
    main()