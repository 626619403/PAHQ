import csv
import random
from prompts import docstring_induction_prompt_generator

def generate_docstring_dataset_csv(num_examples=100, dataset_version='random_doc', output_file='docstring_dataset.csv'):
    """
    使用docstring_induction_prompt_generator生成数据集并保存为CSV文件
    
    参数:
    - num_examples: 生成的示例数量
    - dataset_version: 用于corrupted列的版本名称
    - output_file: 输出CSV文件的路径
    """
    # 设置docstring_induction_prompt_generator的参数
    docstring_ind_prompt_kwargs = {
        'n_matching_args': 3, 
        'n_def_prefix_args': 2, 
        'n_def_suffix_args': 1, 
        'n_doc_prefix_args': 0, 
        'met_desc_len': 3, 
        'arg_desc_len': 2
    }
    
    # 生成原始提示
    print(f"正在生成 {num_examples*2} 个docstring示例...")
    raw_prompts = [
        docstring_induction_prompt_generator("rest", **docstring_ind_prompt_kwargs, seed=i)
        for i in range(num_examples * 2)
    ]
    
    # 准备CSV数据
    csv_data = []
    
    for i, prompt in enumerate(raw_prompts):
        # 提取clean prompt
        clean = prompt.clean_prompt
        
        # 从corrupt_prompt字典中选择指定版本作为corrupted
        corrupted = prompt.corrupt_prompt[dataset_version]
        
        # 从corrupt_prompt字典中选择另一种作为corrupted_hard
        corrupted_hard_key = 'random_def_doc'  # 选择另一个版本
        corrupted_hard = prompt.corrupt_prompt[corrupted_hard_key]
        
        # 添加到数据集
        csv_data.append({
            'clean': clean,
            'corrupted': corrupted,
            'corrupted_hard': corrupted_hard
        })
    
    # 将数据保存到CSV文件
    print(f"正在将数据保存到 {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['clean', 'corrupted', 'corrupted_hard']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)
    
    print(f"成功生成 {len(csv_data)} 条数据并保存到 {output_file}")
    return csv_data

# 生成数据集并保存为CSV
dataset = generate_docstring_dataset_csv(num_examples=100, dataset_version='random_doc', output_file='docstring_dataset.csv')

# 打印前3个示例
print("\n生成的数据集示例:")
for i, example in enumerate(dataset[:3]):
    print(f"\n示例 {i+1}:")
    print(f"Clean:\n{example['clean']}")
    print(f"\nCorrupted:\n{example['corrupted']}")
    print(f"\nCorrupted_hard:\n{example['corrupted_hard']}")
    print("-" * 80)
