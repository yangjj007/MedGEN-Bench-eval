# VLM + 生成/编辑模型

from api.get_vlm_res import *
from api.get_generate_res import *
from api.get_edit_res import *

from util.format_parser import extract_json

import json5
import re
'''
def extract_json(select_response):
    select_response = select_response.replace('```json', '').replace('```', '')
    start_index = select_response.find('{')
    end_index = select_response.rfind('}')
    if start_index != -1 and end_index != -1 and start_index < end_index:
        json_str = select_response[start_index:end_index + 1]
        return json.loads(json_str)
    else:
        return json.loads(select_response)
'''

VLM_ONLY_PROMPT = '''
You are an expert AI assistant specialized in medical image analysis and editing tasks.

You will receive a medical image and a text instruction. Your task is to complete the following objective:

**Task 1: Text Analysis**
Analyze the input medical image according to the given instruction and provide a comprehensive text response.

**Input Instruction:**
{instruction}

**Output Requirements:**
Please provide your response in JSON format with the following structure:
- "output_text": Your analytical response based on the medical image and instruction

Ensure your responses are accurate, medically appropriate, and maintain the integrity of the original medical information while implementing the requested changes.
Now response in json format:
'''


VLM2Edit_PROMPT = '''
You are an expert AI assistant specialized in medical image analysis and editing tasks.

You will receive a medical image and a text instruction. Your task is to complete the following two objectives:

**Task 1: Text Analysis**
Analyze the input medical image according to the given instruction and provide a comprehensive text response.

**Task 2: Image Editing Guidance**
Create a detailed prompt to guide an image editing model in modifying the input medical image according to the specified instruction while preserving relevant medical context and anatomical accuracy.

**Input Instruction:**
{instruction}

**Output Requirements:**
Please provide your response in JSON format with the following structure:
- "output_text": Your analytical response based on the medical image and instruction
- "edit_prompt": A detailed prompt for the image editing model to modify the input image

Ensure your responses are accurate, medically appropriate, and maintain the integrity of the original medical information while implementing the requested changes.
Now response in json format:
'''

# ONLY_EDIT_PROMPT = '''You are an expert AI assistant specialized in medical image editing.

# You will receive a medical image and a text instruction.
# Your task is to modify the input medical image directly according to the given instruction, ensuring that all changes are medically appropriate, anatomically accurate, and preserve the clinical context of the original image.
# Focus on generating an edited version of the image that accurately reflects the requested modification.'''

VLM2Generate_PROMPT = '''
You are an expert AI assistant specialized in medical image analysis and generation tasks.

You will receive a medical image and a text instruction. Your task is to complete the following two objectives:

**Task 1: Text Analysis**
Analyze the input medical image according to the given instruction and provide a comprehensive text response.

**Task 2: Image Generation Guidance**
Create a detailed prompt to guide an image generation model in producing a target image that adheres to both the input instruction and the characteristics of the provided medical image.

**Input Instruction:**
{instruction}

**Output Requirements:**
Please provide your response in JSON format with the following structure:
- "output_text": Your analytical response based on the medical image and instruction
- "generate_prompt": A detailed prompt for the image generation model

Ensure your responses are accurate, medically appropriate, and follow professional standards. Now response in json format:
'''

# todo:实现2个方法，VLM2Generate和VLM2Edit，参考下面的参数列表和对应的usage。注意要支持批量处理，输入和返回均是列表。并且实现一个简单的功能测试函数
'''参数列表
输入参数：
instruction:benchmark中的问题
input_image:benchmark中的问题对应图片保存路径

返回参数
response:模型生成经过解析的文本回答
output_image:模型生成图片保存路径
raw_response:模型文本回答原文

生成图片存./output_image文件夹中
'''


'''vlm usage
async def demo_batch():
    """
    批量处理演示
    """
    client = single_image_vlm(config_path="./api/config.yaml")
    
    # 准备批量请求
    requests = [
        ("请描述这张图片", "./output-old/0c04ad554e0d0c1b.png", 0.7, 2048),
        ("分析这张图片的内容", "./output-old/1e5f5c7214ee52ae_input.jpg", 0.8, 2048),
        ("这张图片展示了什么？", "./output-old/1e5f5c7214ee52ae_input.jpg", 0.6, 2048),
    ]
    
    # 批量处理
    results = await client.generate_batch(requests, concurrency=3)
    
    for i, result in enumerate(results):
        print(f"\n--- 结果 {i+1} ---")
        if "error" in result:
            print(f"错误: {result['error']}")
        else:
            print(result["text"])
            print(f"使用情况: {result['usage']}")


if __name__ == "__main__":
    
    # 批量请求演示
    asyncio.run(demo_batch())
'''

'''image generate usage

    async def generate_images_batch(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """批量生成图片"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(request_data: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    start_time = time.time()
                    paths = await self.generate_image(**request_data)
                    end_time = time.time()
                    
                    return {
                        "request": request_data,
                        "success": True,
                        "paths": paths,
                        "duration": end_time - start_time,
                        "error": None
                    }
                except Exception as e:
                    return {
                        "request": request_data,
                        "success": False,
                        "paths": [],
                        "duration": 0,
                        "error": str(e)
                    }
        
        # 并发执行所有请求
        tasks = [generate_single(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "request": {},
                    "success": False,
                    "paths": [],
                    "duration": 0,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def list_models(self) -> Dict[str, str]:
        """列出所有支持的模型"""
        return {name: config.model_path for name, config in self.model_configs.items()}

# 调试工具
async def debug_single_model(model_name: str = "gpt-image-1", prompt: str = "A simple red apple"):
    """调试单个模型的响应格式"""
    print(f"🔍 调试模型: {model_name}")
    print(f"📝 提示词: {prompt}")
    print("=" * 50)
    
    async with ImageGenerationAPI(debug=True) as api:
        try:
            paths = await api.generate_image(
                model_name=model_name,
                prompt=prompt,
                output_dir="./debug_output",
                file_prefix=f"debug_{model_name}",
                num_images=1
            )
            
            print(f"\n🎯 最终结果:")
            if paths:
                print(f"✅ 成功生成 {len(paths)} 张图片:")
                for path in paths:
                    print(f"  📁 {path}")
            else:
                print("❌ 未生成任何图片")
                
        except Exception as e:
            print(f"❌ 调试失败: {e}")
            import traceback
            traceback.print_exc()


# 完整测试
async def full_test():
    """完整测试所有模型"""
    print("🧪 开始完整测试...")
    
    async with ImageGenerationAPI(debug=False) as api:
        print(f"\n📋 支持的模型 ({len(api.model_configs)} 个):")
        for name, path in api.list_models().items():
            print(f"  - {name}: {path}")
        
        # 测试所有模型
        test_cases = [
            # ("gpt-image-1", "A beautiful red rose"),
            # ("dall-e-3", "A futuristic city"),
            # ("qwen-image", "一只可爱的小猫"),
            # ("ideogram-v3", "Abstract art with geometric shapes"),
            # ("irag-1.0", "一只可爱的小猫"),
            ("doubao-seedream", "一只可爱的小猫"),
            ("imagen-4.0-fast", "一只可爱的小猫"),
        ]
        
        for model_name, prompt in test_cases:
            try:
                print(f"\n📸 测试: {model_name}")
                print(f"   提示词: {prompt}")
                
                start_time = time.time()
                paths = await api.generate_image(
                    model_name=model_name,
                    prompt=prompt,
                    output_dir="./test_output",
                    file_prefix=f"test_{model_name}",
                    num_images=1
                )
                end_time = time.time()
                
                if paths:
                    print(f"✅ 成功 ({end_time-start_time:.1f}s): {paths}")
                else:
                    print("❌ 未生成图片")
                    
            except Exception as e:
                print(f"❌ 失败: {e}")


async def main():
    """主函数"""
    print("🚀 图片生成API工具")
    print("=" * 50)
    
    # 选择运行模式
    print("选择模式:")
    print("1. 调试特定模型")
    print("2. 完整测试")
    print("3. 快速测试")
    
    mode = input("请输入选择 (1/2/3): ").strip()
    
    if mode == "1":
        model_name = input("输入要调试的模型名 (如: gpt-image-1): ").strip()
        if not model_name:
            model_name = "gpt-image-1"
        await debug_single_model(model_name)
    elif mode == "2":
        await full_test()
    else:
        # 快速测试gpt-image-1
        await debug_single_model("gpt-image-1", "A cute kitten playing with a ball")


if __name__ == "__main__":
    asyncio.run(main())
'''

'''image edit usage

    def edit_images_batch(
        self,
        image_prompts: List[Tuple[str, str]],  # [(image_path, prompt), ...]
        save_dir: str = "./output",
        max_concurrent: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """批量处理图片编辑（并发安全）"""
        self._ensure_dir(save_dir)
        results = []
        
        print(f"[{self.model_name}] 开始批量处理 {len(image_prompts)} 个图片编辑任务...")
        
        for i, (image_path, prompt) in enumerate(image_prompts):
            print(f"处理任务 {i+1}/{len(image_prompts)}: {os.path.basename(image_path)}")
            
            try:
                # 为每个任务创建独立的子目录，避免文件名冲突
                task_dir = os.path.join(save_dir, f"task_{i:04d}")
                result = self.edit_image(image_path, prompt, task_dir, **kwargs)
                result["task_index"] = i
                result["image_path"] = image_path
                result["prompt"] = prompt
                results.append(result)
                
            except Exception as e:
                error_result = {
                    "task_index": i,
                    "image_path": image_path,
                    "prompt": prompt,
                    "error": str(e),
                    "text": [],
                    "image_paths": [],
                    "usage": {}
                }
                results.append(error_result)
                print(f"任务 {i+1} 失败: {e}")
        
        print(f"批量处理完成，成功: {sum(1 for r in results if 'error' not in r)}/{len(results)}")
        return results

    async def edit_images_batch_async(
        self,
        image_prompts: List[Tuple[str, str]],
        save_dir: str = "./output",
        max_concurrent: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """异步批量处理图片编辑（并发安全）"""
        self._ensure_dir(save_dir)
        
        async def process_single(i: int, image_path: str, prompt: str):
            try:
                # 为每个任务创建独立的子目录
                task_dir = os.path.join(save_dir, f"task_{i:04d}")
                
                # 注意：这里仍然使用同步方法，如果需要真正的异步，需要改造底层API调用
                result = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.edit_image(image_path, prompt, task_dir, **kwargs)
                )
                result["task_index"] = i
                result["image_path"] = image_path
                result["prompt"] = prompt
                return result
                
            except Exception as e:
                return {
                    "task_index": i,
                    "image_path": image_path,
                    "prompt": prompt,
                    "error": str(e),
                    "text": [],
                    "image_paths": [],
                    "usage": {}
                }
        
        print(f"[{self.model_name}] 开始异步批量处理 {len(image_prompts)} 个任务...")
        
        # 使用信号量控制并发数
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_process(i, image_path, prompt):
            async with semaphore:
                return await process_single(i, image_path, prompt)
        
        # 创建所有任务
        tasks = [
            bounded_process(i, image_path, prompt)
            for i, (image_path, prompt) in enumerate(image_prompts)
        ]
        
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append({
                    "task_index": i,
                    "image_path": image_prompts[i][0],
                    "prompt": image_prompts[i][1],
                    "error": str(result),
                    "text": [],
                    "image_paths": [],
                    "usage": {}
                })
            else:
                final_results.append(result)
        
        success_count = sum(1 for r in final_results if 'error' not in r)
        print(f"异步批量处理完成，成功: {success_count}/{len(final_results)}")
        return final_results
'''
import asyncio
import os
import json
from typing import List, Dict, Any, Tuple
import uuid

async def VLM(
    instructions: List[str],
    input_images: List[str],
    vlm_model: str,
) -> List[Dict[str, Any]]:

    """
    VLM 
    
    Args:
        instructions: 指令列表
        input_images: 输入图片路径列表
    
    Returns:
        List[Dict]: 包含 response, raw_response 的字典列表
    """

    vlm_client = single_image_vlm(config_path="./api/config.yaml", model_name=vlm_model)
    vlm_requests = [
        (VLM_ONLY_PROMPT.format(instruction=instruction), image_path, 0.7, 2048)
        for instruction, image_path in zip(instructions, input_images)
    ]


    # VLM 批量处理
    vlm_results = await vlm_client.generate_batch(vlm_requests, concurrency=5)
    # print("djkdkkddkkkkkkkkkkkkkkkkkkkkkkk",vlm_results)
    results = []
    # generate_requests = []

    # 解析 VLM 结果并准备生成请求
    for i, (vlm_result, instruction, input_image) in enumerate(zip(vlm_results, instructions, input_images)):
        try:
            if "error" in vlm_result:
                results.append({
                    "response": f"VLM Error: {vlm_result['error']}",
                    # "output_image": "",
                    "raw_response": vlm_result.get('text', '')
                })
                continue

            # 解析 VLM 的 JSON 响应
            vlm_response = vlm_result['text']
            parsed_vlm = extract_json(vlm_response)

            # print("vlm返回的 prompt:", parsed_vlm.get("generate_prompt", ""))
            # print("vlm_responsekkkkkkkkkkkkkkkkkkkkkkkkkkkk:", vlm_response)

            output_text = parsed_vlm.get('output_text', '')
            # generate_prompt = parsed_vlm.get('generate_prompt', '')

            # # 准备图像生成请求
            # file_prefix = f"generated_{uuid.uuid4().hex[:8]}"

            # generate_requests.append({
            #     "model_name": generate_model,
            #     "prompt": generate_prompt,
            #     "output_dir": "./output_image",
            #     "file_prefix": file_prefix,
            #     "num_images": 1
            # })

            # print("deweweweweweweweweweweweweweweweeeeeew",generate_prompt)

            results.append({
                "response": output_text,
                # "output_image": "",
                "raw_response": vlm_response,
                # "generate_prompt": generate_prompt,
                # "file_prefix": file_prefix
            })

        except Exception as e:
            results.append({
                "response": f"Parsing Error: {str(e)}",
                # "output_image": "",
                "raw_response": vlm_result.get('text', '')
            })
    
    
    return results



async def VLM2Generate(
    instructions: List[str],
    input_images: List[str],
    vlm_model: str,
    generate_model: str
) -> List[Dict[str, Any]]:

    """
    VLM + 生成模型组合
    
    Args:
        instructions: 指令列表
        input_images: 输入图片路径列表
    
    Returns:
        List[Dict]: 包含 response, output_image, raw_response 的字典列表
    """
    # 确保输出目录存在
    os.makedirs("./output_image", exist_ok=True)
    # ======================================================
    # 特殊模型分支（仿照 VLM2Edit 的 qwen-image-edit 分支）
    # ======================================================
    if generate_model == "gemini-2.5-flash-image-preview" or generate_model == "Showo_GENERATION" or generate_model == "Showo_EDIT":
        results = []
        generate_requests = []

        # ==============================
        # 模拟 VLM 流程（严格仿真版）
        # ==============================
        for instruction, input_image in zip(instructions, input_images):
            try:
                print("instruction命令:",instruction)
                # 1️⃣ 构造伪 VLM 响应文本
                vlm_response = json.dumps({
                    "output_text": "",                  # 模拟空输出
                    "generate_prompt": instruction      # 直接使用 instruction
                })
                
                vlm_dict = json.loads(vlm_response)
                print("伪造vlm返回的 prompt:", vlm_dict["generate_prompt"])
                
                # 2️⃣ 模仿 extract_json 的解析过程
                parsed_vlm = extract_json(vlm_response)

                # 3️⃣ 提取字段，与原逻辑保持一致
                output_text = parsed_vlm.get("output_text", "")
                generate_prompt = parsed_vlm.get("generate_prompt", "")

                # 4️⃣ 准备生成请求
                file_prefix = f"generated_{uuid.uuid4().hex[:8]}"
                #比else逻辑那里多一个input_image的输入
                generate_requests.append({
                    "model_name": generate_model,
                    "prompt": generate_prompt,
                    "output_dir": "./output_image",
                    "file_prefix": file_prefix,
                    "num_images": 1,
                    "input_image": input_image
                })
                
                # 假设 generate_requests 是你的列表
                for i, req in enumerate(generate_requests):
                    print(f"请求 {i+1} 的 prompt:", req["prompt"])
                    print("cbcbcbcbcbcbb",req["input_image"])


                # 5️⃣ 构造结果条目
                results.append({
                    "response": output_text,
                    "output_image": "",
                    "raw_response": vlm_response,   # 保持字符串形式
                    "generate_prompt": generate_prompt,
                    "file_prefix": file_prefix,
                })

            except Exception as e:
                results.append({
                    "response": f"Parsing Error: {str(e)}",
                    "output_image": "",
                    "raw_response": vlm_response
                })

    else:
        # ========================================
        # 通用 VLM + 生成模型逻辑（原 else 部分）
        # ========================================
        # 准备 VLM 批量请求
        vlm_client = single_image_vlm(config_path="./api/config.yaml", model_name=vlm_model)
        vlm_requests = [
            (VLM2Generate_PROMPT.format(instruction=instruction), image_path, 0.7, 2048)
            for instruction, image_path in zip(instructions, input_images)
        ]

        # VLM 批量处理
        vlm_results = await vlm_client.generate_batch(vlm_requests, concurrency=5)
        print("djkdkkddkkkkkkkkkkkkkkkkkkkkkkk",vlm_results)
        results = []
        generate_requests = []

        # 解析 VLM 结果并准备生成请求
        for i, (vlm_result, instruction, input_image) in enumerate(zip(vlm_results, instructions, input_images)):
            try:
                # print("fffffff")
                if "error" in vlm_result:
                    results.append({
                        "response": f"VLM Error: {vlm_result['error']}",
                        "output_image": "",
                        "raw_response": vlm_result.get('text', '')
                    })
                    continue
                # print("ddddd")
                # 解析 VLM 的 JSON 响应
                vlm_response = vlm_result['text'].strip()

                # 去掉开头 ```json 和结尾 ```
                # if vlm_response.startswith("```json"):
                #     vlm_response = vlm_response[len("```json"):].strip()
                # if vlm_response.endswith("```"):
                #     vlm_response = vlm_response[:-3].strip()
                # print("fffffprint",vlm_response)
                # try:
                #     parsed_vlm = extract_json(vlm_response)
                # except Exception as e:
                #     print("extract_json出错:", e)
                #     parsed_vlm = {}

                try:
                    parsed_vlm = json5.loads(vlm_response)
                except Exception as e:
                    print("⚠️ json5 解析失败:", e)
                    parsed_vlm = {}


                # print("vlm返回的 prompt:", parsed_vlm.get("generate_prompt", ""))
                # print("vlm_responsekkkkkkkkkkkkkkkkkkkkkkkkkkkk:", vlm_response)

                output_text = parsed_vlm.get('output_text', '')
                generate_prompt = parsed_vlm.get('generate_prompt', '')

                # 准备图像生成请求
                file_prefix = f"generated_{uuid.uuid4().hex[:8]}"

                generate_requests.append({
                    "model_name": generate_model,
                    "prompt": generate_prompt,
                    "output_dir": "./output_image",
                    "file_prefix": file_prefix,
                    "num_images": 1
                })

                print("deweweweweweweweweweweweweweweweeeeeew",generate_prompt)

                results.append({
                    "response": output_text,
                    "output_image": "",
                    "raw_response": vlm_response,
                    "generate_prompt": generate_prompt,
                    "file_prefix": file_prefix
                })
                print("----------------------------------------------",results)


            except Exception as e:
                results.append({
                    "response": f"Parsing Error: {str(e)}",
                    "output_image": "",
                    "raw_response": vlm_result.get('text', '')
                })
    
    # 图像生成批量处理
    if generate_requests:
        from api.get_generate_res import ImageGenerationAPI
        async with ImageGenerationAPI() as gen_api:
            for i, req in enumerate(generate_requests):
                print(f"ttttt{i+1} 的 prompt:", req["prompt"])

            gen_results = await gen_api.generate_images_batch(generate_requests, max_concurrent=3)
            
            # 更新结果中的图片路径
            gen_index = 0
            for i, result in enumerate(results):
                if "generate_prompt" in result and gen_index < len(gen_results):
                    gen_result = gen_results[gen_index]
                    if gen_result["success"] and gen_result["paths"]:
                        result["output_image"] = gen_result["paths"][0]
                    else:
                        result["output_image"] = f"Generation Error: {gen_result.get('error', 'Unknown error')}"
                    # 移除临时字段
                    result.pop("generate_prompt", None)
                    result.pop("file_prefix", None)
                    gen_index += 1
    
    return results


async def VLM2Edit(
    instructions: List[str],
    input_images: List[str],
    vlm_model: str,
    edit_model: str
) -> List[Dict[str, Any]]:

    """
    VLM + 编辑模型组合
    
    Args:
        instructions: 指令列表
        input_images: 输入图片路径列表
    
    Returns:
        List[Dict]: 包含 response, output_image, raw_response 的字典列表
    """
    # 确保输出目录存在
    os.makedirs("./output_image", exist_ok=True)

    if edit_model == 'qwen-image-edit' or edit_model == 'doubao-seedream' or edit_model == 'Ming-UniVision_EDIT' or edit_model == 'Ming-UniVision_EDIT4GEN':

        # edit_requests = [
        #     (VLM2Edit_PROMPT.format(instruction=instruction), image_path, 0.7, 2048)
        #     for instruction, image_path in zip(instructions, input_images)
        # ]
        
        results = []
        edit_requests = []

        # ==============================
        # 模拟 VLM 流程（严格仿真版）
        # ==============================
        for instruction, input_image in zip(instructions, input_images):
            try:
                # 1️⃣ 构造伪 VLM 响应文本（与真实完全一致，仅包含 output_text / edit_prompt）
                vlm_response = json.dumps({
                    "output_text": "",              # 模拟空的 VLM 输出
                    "edit_prompt": instruction      # 直接使用 instruction
                })

                # 2️⃣ 模仿 extract_json 的解析过程
                parsed_vlm = extract_json(vlm_response)

                # 3️⃣ 提取字段，与原逻辑完全一致
                output_text = parsed_vlm.get("output_text", "")
                edit_prompt = parsed_vlm.get("edit_prompt", "")

                # 4️⃣ 准备编辑请求
                edit_requests.append((input_image, edit_prompt))

                # 5️⃣ 构造结果条目，raw_response 为字符串（与真实 VLM 保持一致）
                results.append({
                    "response": output_text,
                    "output_image": "",  # 稍后填充
                    "raw_response": vlm_response,   # ⚠️ 注意，这里是字符串，不是 dict
                    "edit_prompt": edit_prompt,
                    "original_image": input_image
                })

            except Exception as e:
                results.append({
                    "response": f"Parsing Error: {str(e)}",
                    "output_image": "",
                    "raw_response": vlm_response
                })

    else:
        # 准备 VLM 批量请求
        vlm_client = single_image_vlm(config_path="./api/config.yaml", model_name=vlm_model)
        
        vlm_requests = [
            (VLM2Edit_PROMPT.format(instruction=instruction), image_path, 0.7, 2048)
            for instruction, image_path in zip(instructions, input_images)
        ]
        
        # VLM 批量处理
        vlm_results = await vlm_client.generate_batch(vlm_requests, concurrency=5)
        
        results = []
        edit_requests = []
        
        # 解析 VLM 结果并准备编辑请求
        for i, (vlm_result, instruction, input_image) in enumerate(zip(vlm_results, instructions, input_images)):
            try:
                if "error" in vlm_result:
                    results.append({
                        "response": f"VLM Error: {vlm_result['error']}",
                        "output_image": "",
                        "raw_response": vlm_result.get('text', '')
                    })
                    continue
                
                # 解析 VLM 的 JSON 响应
                vlm_response = vlm_result['text']
                parsed_vlm = extract_json(vlm_response)
                
                output_text = parsed_vlm.get('output_text', '')
                edit_prompt = parsed_vlm.get('edit_prompt', '')
                
                # 准备图像编辑请求
                edit_requests.append((input_image, edit_prompt))
                
                results.append({
                    "response": output_text,
                    "output_image": "",  # 稍后填充
                    "raw_response": vlm_response,
                    "edit_prompt": edit_prompt,
                    "original_image": input_image
                })
                
            except Exception as e:
                results.append({
                    "response": f"Parsing Error: {str(e)}",
                    "output_image": "",
                    "raw_response": vlm_result.get('text', '')
                })
    
    # 图像编辑批量处理
    if edit_requests:
        from api.get_edit_res import ImageEditAPI  # 根据实际API类名调整
        edit_api = ImageEditAPI(model_name=edit_model)  # 根据实际API初始化方式调整
        
        if edit_model == 'Ming-UniVision_EDIT' or edit_model == 'Ming-UniVision_EDIT4GEN':
            print("qwertyuoooooooooooooooooooo")
            edit_results = edit_api.edit_images_batch(
                edit_requests, 
                save_dir="./output_image",
                max_concurrent=1
            )
        else:
            try:
                # 使用异步批量编辑
                edit_results = await edit_api.edit_images_batch_async(
                    edit_requests, 
                    save_dir="./output_image",
                    max_concurrent=3
                )
            except AttributeError:
                # 如果没有异步方法，使用同步方法
                edit_results = edit_api.edit_images_batch(
                    edit_requests, 
                    save_dir="./output_image",
                    max_concurrent=3
                )
        
        # 更新结果中的图片路径
        edit_index = 0
        for i, result in enumerate(results):
            if "edit_prompt" in result and edit_index < len(edit_results):
                edit_result = edit_results[edit_index]
                if "error" not in edit_result and edit_result.get("image_paths"):
                    result["output_image"] = edit_result["image_paths"][0]
                else:
                    result["output_image"] = f"Edit Error: {edit_result.get('error', 'Unknown error')}"
                # 移除临时字段
                result.pop("edit_prompt", None)
                result.pop("original_image", None)
                edit_index += 1
    
    return results


def test_functionality(vlm_model: str, edit_model: str, generate_model: str):
    """简单的功能测试函数"""
    async def run_tests():
        print("🧪 开始功能测试...")
        print("=" * 50)
        
        # 测试数据 - 你需要准备实际的测试图片
        test_instructions = [
            "Generate 3D reconstruction from this low attenuation CT slice",
            "Annotate this CT image of urinary tract with detailed labels for the absence of abnormal areas of enhancement, including anatomical landmarks and diagnostic findings", 
        ]
        
        test_images = [
            "/data2/wangchangmiao/yjj/medical_AI/medical-bench-eval/MedGEN/images/gen/concat_labeled_grid_s0061.jpg",
            "/data2/wangchangmiao/yjj/medical_AI/medical-bench-eval/MedGEN/images/edit/498db349db622e828f29e7ee9d438367.jpg"
        ]
        
        # 检查测试图片是否存在
        available_images = []
        available_instructions = []
        
        for i, img_path in enumerate(test_images):
            if os.path.exists(img_path):
                available_images.append(img_path)
                available_instructions.append(test_instructions[i])
            else:
                print(f"⚠️  测试图片不存在: {img_path}")
        
        if not available_images:
            print("❌ 没有找到测试图片，请准备测试图片后再运行")
            print("请在以下位置放置测试图片：")
            for img_path in test_images:
                print(f"  - {img_path}")
            return
        
        print(f"✅ 找到 {len(available_images)} 张测试图片")
        
        try:
            # 测试 VLM2Generate
            print("\n📸 测试 VLM2Generate...")
            print("-" * 30)
            
            generate_results = await VLM2Generate(
                available_instructions[:1], 
                available_images[:1],
                vlm_model,
                generate_model
            )
            
            for i, result in enumerate(generate_results):
                print(f"\n生成结果 {i+1}:")
                print(f"  指令: {available_instructions[i][:50]}...")
                print(f"  文本回答: {result['response'][:100]}...")
                print(f"  生成图片: {result['output_image']}")
                
                # 检查生成是否成功
                success = (result['output_image'] and 
                          os.path.exists(result['output_image']) and 
                          not result['output_image'].startswith('Generation Error'))
                print(f"  生成状态: {'✅ 成功' if success else '❌ 失败'}")
                
                if success:
                    file_size = os.path.getsize(result['output_image'])
                    print(f"  文件大小: {file_size / 1024:.1f} KB")
            
            # 测试 VLM2Edit
            print("\n🎨 测试 VLM2Edit...")
            print("-" * 30)
            
            edit_results = await VLM2Edit(
                available_instructions[:1], 
                available_images[:1],
                vlm_model,
                edit_model
            )
            
            for i, result in enumerate(edit_results):
                print(f"\n编辑结果 {i+1}:")
                print(f"  指令: {available_instructions[i][:50]}...")
                print(f"  文本回答: {result['response'][:100]}...")
                print(f"  编辑图片: {result['output_image']}")
                
                # 检查编辑是否成功
                success = (result['output_image'] and 
                          os.path.exists(result['output_image']) and 
                          not result['output_image'].startswith('Edit Error'))
                print(f"  编辑状态: {'✅ 成功' if success else '❌ 失败'}")
                
                if success:
                    file_size = os.path.getsize(result['output_image'])
                    print(f"  文件大小: {file_size / 1024:.1f} KB")
            
            # 批量测试
            if len(available_images) > 1:
                print("\n🔄 测试批量处理...")
                print("-" * 30)
                
                batch_results = await VLM2Generate(available_instructions, available_images)
                success_count = sum(1 for r in batch_results 
                                  if r['output_image'] and os.path.exists(r['output_image']))
                print(f"批量生成测试: {success_count}/{len(batch_results)} 成功")
                
                batch_results = await VLM2Edit(available_instructions, available_images)
                success_count = sum(1 for r in batch_results 
                                  if r['output_image'] and os.path.exists(r['output_image']))
                print(f"批量编辑测试: {success_count}/{len(batch_results)} 成功")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 50)
        print("✅ 功能测试完成")
        
        # 显示输出目录信息
        if os.path.exists("./output_image"):
            files = os.listdir("./output_image")
            print(f"\n📁 输出目录包含 {len(files)} 个文件:")
            for file in files[:5]:  # 只显示前5个文件
                file_path = os.path.join("./output_image", file)
                file_size = os.path.getsize(file_path) / 1024
                print(f"  - {file} ({file_size:.1f} KB)")
            if len(files) > 5:
                print(f"  ... 还有 {len(files) - 5} 个文件")
    
    # 运行测试
    asyncio.run(run_tests())


# 额外的工具函数
async def batch_process_demo():
    """批量处理演示"""
    print("🚀 批量处理演示")
    
    # 示例：处理多个医学图片
    instructions = [
        # "Generate 3D reconstruction from this low attenuation CT slice",
        # "Annotate this CT image of urinary tract with detailed labels for the absence of abnormal areas of enhancement, including anatomical landmarks and diagnostic findings", 
        "基于这张图生成翻转视角的图像",
        "增强图片的对比度"
    ]
    
    images = [
        "/data2/wangchangmiao/yjj/medical_AI/medical-bench-eval/MedGEN/images/gen/concat_labeled_grid_s0061.jpg",
        "/data2/wangchangmiao/yjj/medical_AI/medical-bench-eval/MedGEN/images/edit/498db349db622e828f29e7ee9d438367.jpg"
    ]
    
    # 检查图片是否存在
    valid_pairs = [(inst, img) for inst, img in zip(instructions, images) if os.path.exists(img)]
    
    if not valid_pairs:
        print("❌ 没有找到示例图片")
        return
    
    valid_instructions, valid_images = zip(*valid_pairs)
    
    print(f"处理 {len(valid_pairs)} 个图片...")
    
    # 同时运行生成和编辑
    generate_task = VLM2Generate(list(valid_instructions), list(valid_images))
    edit_task = VLM2Edit(list(valid_instructions), list(valid_images))
    
    generate_results, edit_results = await asyncio.gather(generate_task, edit_task)
    
    print("📊 批量处理结果:")
    print(f"生成任务: {len([r for r in generate_results if os.path.exists(r.get('output_image', ''))])}/{len(generate_results)} 成功")
    print(f"编辑任务: {len([r for r in edit_results if os.path.exists(r.get('output_image', ''))])}/{len(edit_results)} 成功")


if __name__ == "__main__":
    
    vlm_model = 'qwen3-vl-235b-a22b-instruct'
    generate_model = 'dall-e-3'
    edit_model = 'qwen-image-edit'
    
    
    print("🔧 VLM + 生成/编辑模型工具")
    print("选择运行模式:")
    print("1. 功能测试")
    print("2. 批量处理演示")
    
    choice = input("请选择 (1/2): ").strip()
    
    if choice == "1":
        test_functionality(vlm_model,edit_model,generate_model)
    elif choice == "2":
        asyncio.run(batch_process_demo())
    else:
        print("默认运行功能测试...")
        test_functionality(vlm_model,edit_model,generate_model)