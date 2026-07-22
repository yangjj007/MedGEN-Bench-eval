# diffusion类文本生成图片模型api

# 图片生成和编辑api在高并发情况下，多个图片返回储存到同一个输出路径的同一个图片文件内，会导致竞争条件。解决方法就是在输出路径创建若干个临时文件夹，保存到里面再移动到输出路径，最后删除临时文件夹。请你修改我的代码，给其中的批量回答方法加入上述机制。给我需要修改的代码部分就可以，不必全部输出
# 使用这两个库实现：shutil uuid

import yaml
import aiohttp
import asyncio
import base64
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import json
import time
from pathlib import Path
import logging
import requests

import uuid
import shutil

from openai import OpenAI, AsyncOpenAI

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """图片生成配置"""
    prompt: str
    output_dir: str = "."
    file_prefix: str = "generated_image"
    extra_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelConfig:
    """模型配置"""
    model_path: str
    default_params: Dict[str, Any]
    param_mapping: Dict[str, str]  # 通用参数到模型特定参数的映射

class ImageGenerationAPI:
    """通用图片生成API调用类"""
    
    def __init__(self, config_path: str = "./api/config.yaml", debug: bool = False):
        """初始化API客户端"""
        self.config = self._load_config(config_path)
        self._validate_config()
        self.session = None
        self.model_configs = self._init_model_configs()
        self.debug = debug
        self.client = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"配置文件未找到: {config_path}")
            self._create_default_config(config_path)
            raise
        except yaml.YAMLError as e:
            logger.error(f"配置文件格式错误: {e}")
            raise
    
    def _create_default_config(self, config_path: str):
        """创建默认配置文件"""
        default_config = {
            'api_key': "sk-your-api-key-here",
            'base_url': "https://aihubmix.com/v1",
            'model_name': "",
            'temperature': 0.4,
            'site_url': "",
            'site_name': "",
            'max_retries': 1,
            'retry_delay': 2
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"已创建默认配置文件: {config_path}")
        logger.info("请在配置文件中填入你的真实API密钥！")
    
    def _validate_config(self):
        """验证配置"""
        required_fields = ['api_key', 'base_url']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"配置文件缺少必需字段: {field}")
        
        # 检查API密钥是否为默认值
        if self.config['api_key'] in ["sk-test-key", "sk-your-api-key-here", ""]:
            logger.error("❌ API密钥无效！")
            logger.error("请在配置文件 './api/config.yaml' 中填入你的真实API密钥")
            logger.error("API密钥格式应该类似: sk-xxxxxxxxxxxxxxxx")
            raise ValueError("请配置有效的API密钥")
        
        logger.info(f"✅ 配置验证通过 - API密钥: {self.config['api_key'][:8]}...")
    
    def _init_model_configs(self) -> Dict[str, ModelConfig]:
        """初始化模型配置映射表"""
        return {
            # OpenAI GPT Image系列
            "gpt-image-1": ModelConfig(
                model_path="opanai/gpt-image-1",
                default_params={
                    "size": "1024x1024",
                    "n": 1,
                    "quality": "high",
                    "moderation": "low",
                    "background": "auto"
                },
                param_mapping={
                    "num_images": "n",
                    "image_size": "size",
                    "image_quality": "quality"
                }
            ),
            
            "dall-e-3": ModelConfig(
                model_path="opanai/dall-e-3",
                default_params={
                    "size": "1024x1024",
                    "n": 1
                },
                param_mapping={
                    "num_images": "n",
                    "image_size": "size"
                }
            ),
            

            # Google Imagen
            "imagen-4.0-fast": ModelConfig(
                model_path="google/imagen-4.0-fast-generate-001",
                default_params={
                    "numberOfImages": 1
                },
                param_mapping={
                    "num_images": "numberOfImages"
                }
            ),

            "gemini-2.5-flash-image-preview": ModelConfig(
                model_path="google/gemini-2.5-flash-image-preview",
                default_params={
                    "modalities": ["text", "image"],
                    "temperature": 0.7,
                    "max_tokens": 16384
                },
                param_mapping={
                }
            ),
            
            # 千帆系列
            "qwen-image": ModelConfig(
                model_path="qianfan/qwen-image",
                default_params={
                    "refer_image": "",
                    "n": 1,
                    "size": "1024x1024",
                    "guidance": 7.5,
                    "watermark": False
                },
                param_mapping={
                    "num_images": "n",
                    "image_size": "size",
                    "guidance_scale": "guidance"
                }
            ),
            
            "irag-1.0": ModelConfig(
                model_path="qianfan/irag-1.0",
                default_params={
                    "refer_image": "",
                    "n": 1,
                    "size": "1024x1024",
                    "guidance": 7.5,
                    "watermark": False
                },
                param_mapping={
                    "num_images": "n",
                    "image_size": "size",
                    "guidance_scale": "guidance"
                }
            ),
            
            # 豆包系列
            "doubao-seedream": ModelConfig(
                model_path="doubao/doubao-seedream-4-0-250828",
                default_params={
                    "size": "2K",
                    "sequential_image_generation": "disabled",
                    "stream": False,
                    "response_format": "url",
                    "watermark": False
                },
                param_mapping={
                    "image_size": "size",
                    "return_format": "response_format"
                }
            ),
            
            # Ideogram
            "ideogram-v3": ModelConfig(
                model_path="ideogram/V3",
                default_params={
                    "rendering_speed": "QUALITY",
                    "aspect_ratio": "2x1"
                },
                param_mapping={
                    "quality": "rendering_speed",
                    "aspect": "aspect_ratio"
                }
            ),
            
        }
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        timeout = aiohttp.ClientTimeout(total=300)  # 5分钟超时
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    def _prepare_params(self, model_name: str, **kwargs) -> Dict[str, Any]:
        """准备模型参数"""
        if model_name not in self.model_configs:
            available_models = list(self.model_configs.keys())
            raise ValueError(f"不支持的模型: {model_name}. 可用模型: {available_models}")
        
        model_config = self.model_configs[model_name]
        params = model_config.default_params.copy()
        
        # 应用参数映射
        for common_param, model_param in model_config.param_mapping.items():
            if common_param in kwargs:
                params[model_param] = kwargs[common_param]
                kwargs.pop(common_param)
        
        # 直接覆盖其他参数
        params.update(kwargs)
        
        return params
    
    async def _make_request(self, model_name: str, prompt: str, **kwargs) -> Dict[str, Any]:
        
        print("cccccccccprompt:", prompt)
        
        """发起API请求"""
        if not self.session:
            raise RuntimeError("请在异步上下文中使用此方法")
        
        model_config = self.model_configs[model_name]

        # if model_name == 'ideogram-v3':
        #     url = "https://aihubmix.com/ideogram/v1/ideogram-v3/generate"
        # else:
        #     url = f"{self.config['base_url']}/models/{model_config.model_path}/predictions"

        url = f"{self.config['base_url']}/models/{model_config.model_path}/predictions"

        # if model_name == 'ideogram-v3':
        #     headers = {
        #         "Api-Key": f"{self.config['api_key']}"
        #     }
        # else:
        #     headers = {
        #         "Content-Type": "application/json",
        #         "Authorization": f"Bearer {self.config['api_key']}"
        #     }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config['api_key']}"
        }

        params = self._prepare_params(model_name, **kwargs)
        params["prompt"] = prompt

        print("dddddddprompt:", params["prompt"])

        # 从 kwargs 拿 image，如果没有也不会报错
        input_image = kwargs.get("input_image")
        
        print("wewewewwwwwwwwwwwww",input_image)
        
        if model_name == "gemini-2.5-flash-image-preview":
            
            self.client = OpenAI(
                api_key=self.config["api_key"],
                base_url=self.config.get("base_url", "https://aihubmix.com/v1"),
            )

            image_path = input_image
            def encode_image(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")
            base64_image = encode_image(image_path)

            response = self.client.chat.completions.create(
                model="gemini-2.5-flash-image-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url", 
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },     
                        ],
                    },
                ],
                modalities=["text", "image"],
                temperature=0.7,
            )

            try:
                # 直接取第一个 choice -> message -> multi_mod_content -> inline_data -> data
                # base64_str = response.choices[0].message.multi_mod_content[0].inline_data.data
                # base64_str = response["choices"][0]["message"]["multi_mod_content"][0]["inline_data"]["data"]
                base64_str = response.choices[0].message.multi_mod_content[0]["inline_data"]["data"]

                # 打印前50个字符调试
                if base64_str:
                    print("DEBUG: base64_str 前50字符:", base64_str[:50])
                else:
                    print("DEBUG: base64_str 为空或 None")

                return base64_str
            except Exception as e:
                print(f"获取 base64 图片失败: {e}")
                return None

            # try:
            #     results = {
            #         "created": getattr(response, "created", None),
            #         "usage": {
            #             "completion_tokens": getattr(response.usage, "completion_tokens", None),
            #             "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            #             "total_tokens": getattr(response.usage, "total_tokens", None),
            #         },
            #         "texts": [],
            #         "images": []
            #     }

            #     choices = getattr(response, "choices", [])
            #     if choices:
            #         message = getattr(choices[0], "message", {})
            #         multi_mod_content = getattr(message, "multi_mod_content", [])
            #         for part in multi_mod_content:
            #             # 文本
            #             if "text" in part and part["text"]:
            #                 results["texts"].append(part["text"])

            #             # 图片
            #             elif "inline_data" in part and part["inline_data"]:
            #                 inline = part["inline_data"]
            #                 img_b64 = inline.get("data")
            #                 mime_type = inline.get("mime_type", "image/png")
            #                 if img_b64:
            #                     results["images"].append({
            #                         "type": "base64",
            #                         "data": img_b64,
            #                         "mimeType": mime_type
            #                     })

            #     return results
            
            # except Exception as e:
            #     # 出错时返回 error 字段
            #     return {"error": str(e)}

            # import json

            # def safe_truncate_base64(data, max_len=50):
            #     """截断长字符串，仅显示前max_len个字符"""
            #     if isinstance(data, str) and len(data) > max_len:
            #         if data.startswith("iVBOR") or data.startswith("/9j/"):  # 常见的base64开头
            #             return data[:max_len] + "...(base64 truncated)"
            #         return data
            #     return data


            # def safe_json(obj, max_len=50):
            #     """递归地将 response 对象转换为安全可读结构"""
            #     if isinstance(obj, dict):
            #         return {k: safe_json(v, max_len) for k, v in obj.items()}
            #     elif isinstance(obj, list):
            #         return [safe_json(v, max_len) for v in obj]
            #     elif hasattr(obj, "__dict__"):
            #         return safe_json(vars(obj), max_len)
            #     else:
            #         return safe_truncate_base64(obj, max_len)


            # # === 使用方式 ===
            # try:
            #     # 如果是OpenAI官方SDK的响应对象
            #     response_dict = (
            #         response.model_dump() if hasattr(response, "model_dump") else response.to_dict()
            #     )
                
            #     safe_response = safe_json(response_dict, max_len=80)
            #     print("iiiiiiiiiiiiiiiiiiiii",json.dumps(safe_response, indent=2, ensure_ascii=False),"ddddddddddd")

            # except Exception as e:
            #     print(f"无法安全打印 response: {e}")


            # try:
            #     # 初始化 results 结构，保持与参考一致
            #     results = {
            #         "created": getattr(response, "created", None),
            #         "usage": {"total_tokens": getattr(response.usage, "total_tokens", None)},
            #         "texts": [],
            #         "images": []
            #     }

            #     print(f"Creation time: {results['created']}")
            #     print(f"Token usage: {results['usage']['total_tokens']}")

            #     # 检查 multimodal 内容
            #     if (
            #         hasattr(response.choices[0].message, "multi_mod_content")
            #         and response.choices[0].message.multi_mod_content is not None
            #     ):
            #         print("\nResponse content:")
            #         for part in response.choices[0].message.multi_mod_content:
            #             # 文本内容
            #             if "text" in part and part["text"] is not None:
            #                 text = part["text"]
            #                 results["texts"].append(text)
            #                 print(text)
                        
            #             # 图像内容
            #             elif "inline_data" in part and part["inline_data"] is not None:
            #                 print("\n🖼️ [Image content received]")
            #                 img_b64 = part["inline_data"]["data"]
            #                 mime_type = part["inline_data"].get("mime_type", "image/png")
            #                 print(f"Image type: {mime_type}")
                            
            #                 # 保存图像文件
            #                 image_data = base64.b64decode(img_b64)
            #                 image = Image.open(BytesIO(image_data))
            #                 image.show()

            #                 output_dir = os.path.join(os.path.dirname(image_path), "output")
            #                 os.makedirs(output_dir, exist_ok=True)
            #                 output_path = os.path.join(output_dir, "edited_image.jpg")
            #                 image.save(output_path)
            #                 print(f"✅ Image saved to: {output_path}")
                            
            #                 # 添加到 results.images（与参考结构一致）
            #                 results["images"].append({
            #                     "type": "base64",
            #                     "data": img_b64,
            #                     "mimeType": mime_type
            #                 })
            #     else:
            #         print("No valid multimodal response received, check response structure")
            #         results["error"] = "No valid multimodal response"

            # except Exception as e:
            #     print(f"Error processing response: {str(e)}")
            #     results = {"error": str(e)}

            # # ✅ 最后返回 results
            # return results

        else:
            logger.info(f"发起请求: {model_name} - {url}")
            if self.debug:
                logger.debug(f"请求参数: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            
            for attempt in range(self.config.get('max_retries', 3)):
                try:
                    async with self.session.post(url, json=payload, headers=headers) as response:
                        response_text = await response.text()
                        
                        if response.status == 200:
                            result = json.loads(response_text)
                            logger.info(f"请求成功: {model_name}")
                            
                            # 调试：打印响应结构
                            if self.debug:
                                print("\n🔍 完整响应结构:")
                                print("=" * 50)
                                print(json.dumps(result, indent=2, ensure_ascii=False))
                                print("=" * 50)
                            
                            return result
                        else:
                            logger.error(f"请求失败 (状态码: {response.status}): {response_text}")
                            
                            # 特殊处理常见错误
                            if response.status == 401:
                                logger.error("❌ 认证失败：API密钥无效或已过期")
                            elif response.status == 403:
                                logger.error("❌ 权限不足：账户可能没有访问此模型的权限")
                            elif response.status == 429:
                                logger.error("❌ 请求频率过高：已达到API调用限制")
                            elif response.status == 500:
                                logger.error("❌ 服务器内部错误：模型可能暂时不可用")
                            
                            if attempt < self.config.get('max_retries', 3) - 1:
                                retry_delay = self.config.get('retry_delay', 2)
                                logger.info(f"等待 {retry_delay} 秒后重试...")
                                await asyncio.sleep(retry_delay)
                            else:
                                raise aiohttp.ClientResponseError(
                                    request_info=response.request_info,
                                    history=response.history,
                                    status=response.status,
                                    message=response_text
                                )
                except json.JSONDecodeError as e:
                    logger.error(f"响应JSON解析失败: {e}")
                    if attempt < self.config.get('max_retries', 3) - 1:
                        await asyncio.sleep(self.config.get('retry_delay', 2))
                    else:
                        raise
                except Exception as e:
                    logger.error(f"请求异常 (尝试 {attempt + 1}): {e}")
                    if attempt < self.config.get('max_retries', 3) - 1:
                        await asyncio.sleep(self.config.get('retry_delay', 2))
                    else:
                        raise
    
    def _save_image_from_base64(self, b64_data: str, output_dir: str, file_prefix: str, index: int = 0) -> str:
        """从base64数据保存图片"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理文件名冲突
        current_index = index
        while True:
            file_name = f"{file_prefix}_{current_index}.png"
            file_path = os.path.join(output_dir, file_name)
            
            if not os.path.exists(file_path):
                break
            current_index += 1
        
        # 确保b64_data是字符串
        if not isinstance(b64_data, str):
            raise ValueError(f"base64数据必须是字符串，但收到: {type(b64_data)}")
        
        # 解码并保存图片
        try:
            image_bytes = base64.b64decode(b64_data)
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            
            logger.info(f"图片已保存至: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"base64解码失败: {e}")
            logger.error(f"数据类型: {type(b64_data)}")
            logger.error(f"数据内容预览: {str(b64_data)[:100]}...")
            raise
    
    async def _download_image_from_url(self, url: str, output_dir: str, file_prefix: str, index: int = 0) -> str:
        """从URL下载图片"""
        if not self.session:
            raise RuntimeError("请在异步上下文中使用此方法")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理文件名冲突
        current_index = index
        while True:
            file_name = f"{file_prefix}_{current_index}.png"
            file_path = os.path.join(output_dir, file_name)
            
            if not os.path.exists(file_path):
                break
            current_index += 1
        
        # 下载图片
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    with open(file_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    
                    logger.info(f"图片已下载至: {file_path}")
                    return file_path
                else:
                    error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"下载图片失败: {error_text}"
                    )
        except Exception as e:
            logger.error(f"下载图片失败: {url} - {e}")
            raise
        
    def _extract_images_from_response(self, response: Dict[str, Any], model_name: str) -> List[Dict[str, Any]]:
        """从响应中提取图片数据（最终健壮版）"""
        
        if self.debug:
            print(f"\n🔍 开始提取图片数据...")
        
        found_images = []
        
        def find_images_in_data(data, path=""):
            """递归查找图片数据"""
            
            # --- 1. 如果当前数据是字典 ---
            if isinstance(data, dict):
                # 检查当前字典本身是否代表一个图片对象
                # Case A: {"bytesBase64": "...", "mimeType": "..."}
                if "bytesBase64" in data and isinstance(data.get("bytesBase64"), str):
                    if self.debug: print(f"✅ 找到了Base64图片 (在字典对象 {path} 中)")
                    found_images.append({
                        "type": "base64", "data": data["bytesBase64"],
                        "path": f"{path}.bytesBase64",
                        "mimeType": data.get("mimeType", "png").split('/')[-1]
                    })
                    return # 找到后不再深入此字典
                
                # Case B: {"url": "..."}
                if "url" in data and isinstance(data.get("url"), str) and data["url"].startswith("http"):
                    if self.debug: print(f"✅ 找到了URL图片 (在字典对象 {path} 中)")
                    found_images.append({
                        "type": "url", "data": data["url"], "path": f"{path}.url"
                    })
                    # 不返回，因为可能还有 b64_json
                
                # Case C: {"b64_json": "..."}
                if "b64_json" in data and isinstance(data.get("b64_json"), str):
                    if self.debug: print(f"✅ 找到了Base64图片 (在字典对象 {path} 中)")
                    found_images.append({
                        "type": "base64", "data": data["b64_json"],
                        "path": f"{path}.b64_json", "mimeType": "png"
                    })

                # 无论当前字典是否是图片对象，都继续递归其值
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    find_images_in_data(value, current_path)

            # --- 2. 如果当前数据是列表 ---
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    current_path = f"{path}[{i}]" if path else f"[{i}]"
                    
                    # <<< START: 关键修复点 >>>
                    # Case D: 直接处理列表中的URL字符串
                    if isinstance(item, str) and item.startswith("http"):
                        if self.debug: print(f"✅ 找到了URL图片 (在列表 {path} 中)")
                        found_images.append({
                            "type": "url", "data": item, "path": current_path
                        })
                    # <<< END: 关键修复点 >>>
                    
                    # 否则，继续递归处理列表中的元素（可能是字典或子列表）
                    else:
                        find_images_in_data(item, current_path)

        # --- 开始执行查找 ---
        find_images_in_data(response)
        
        # --- 去重 ---
        unique_images = []
        seen_data = set()
        for img in found_images:
            # 使用数据的前100个字符作为唯一标识符
            identifier = img['data'][:100]
            if identifier not in seen_data:
                unique_images.append(img)
                seen_data.add(identifier)

        if self.debug:
            print(f"\n📊 提取到 {len(unique_images)} 张唯一图片:")
            if unique_images:
                for i, img in enumerate(unique_images):
                    print(f"  {i+1}. 类型: {img['type']:<7} | 路径: {img['path']:<40} | 格式: {img.get('mimeType', 'N/A')}")
            else:
                 print("  (无)")
        
        return unique_images
    
    
    async def generate_image(
        self,
        model_name: str,
        prompt: str,
        output_dir: str = ".",
        file_prefix: str = "generated_image",
        **kwargs
    ) -> List[str]:
        """生成图片"""
        try:
            print("bbbbbbbbbbbprompt:", prompt)

            # === 如果 model_name 是本地 Ming-UniVision 生成 ===
            if model_name == "Ming-UniVision_GENERATION":

                local_api = "http://127.0.0.1:10092/api/run_model"
                os.makedirs(output_dir, exist_ok=True)

                # prompt = "一只可爱的小猫。"

                payload = {
                    "model_name": "Ming-UniVision_GENERATION",
                    "weight_version": "Ming-UniVision-16B-A3B",
                    "mode": "generation",
                    "input_text": prompt,
                    "input_image_base64": None
                }

                # POST form-data 请求本地 API
                response = requests.post(local_api, data=payload, timeout=180)
                response.raise_for_status()
                result = response.json()

                # 拿出返回字段
                output_text = result.get("output_text", "")
                output_img_base64 = result.get("output_image_base64", None)

                saved_paths = []
                if output_img_base64:
                    img_bytes = base64.b64decode(output_img_base64)
                    local_path = os.path.join(output_dir, f"{file_prefix}_{uuid.uuid4().hex}_0.jpg")
                    with open(local_path, "wb") as f:
                        f.write(img_bytes)
                    saved_paths.append(local_path)

                return saved_paths
            
            elif model_name == "Showo_GENERATION":
                local_api = "http://127.0.0.1:10098/api/run_model"
                os.makedirs(output_dir, exist_ok=True)

                # prompt = "一只可爱的小猫。"

                payload = {
                    "model_name": "Showo_GENERATION",
                    "weight_version": "T2I",
                    "mode": "T2I",
                    "input_text": prompt,
                    "input_image_base64": None
                }

                # POST form-data 请求本地 API
                response = requests.post(local_api, data=payload, timeout=180)
                response.raise_for_status()
                result = response.json()

                # 拿出返回字段
                output_text = result.get("output_text", "")
                output_img_base64 = result.get("output_image_base64", None)

                saved_paths = []
                if output_img_base64:
                    img_bytes = base64.b64decode(output_img_base64)
                    local_path = os.path.join(output_dir, f"{file_prefix}_{uuid.uuid4().hex}_0.jpg")
                    with open(local_path, "wb") as f:
                        f.write(img_bytes)
                    saved_paths.append(local_path)

                return saved_paths

            elif model_name == "Showo_EDIT":
                local_api = "http://127.0.0.1:10100/api/run_model"
                os.makedirs(output_dir, exist_ok=True)

                # prompt = "一只可爱的小猫。"

                payload = {
                    "model_name": "Showo_EDIT",
                    "weight_version": "T2I",
                    "mode": "T2I",
                    "input_text": prompt,
                    "input_image_base64": None
                }

                # POST form-data 请求本地 API
                response = requests.post(local_api, data=payload, timeout=180)
                response.raise_for_status()
                result = response.json()

                # 拿出返回字段
                output_text = result.get("output_text", "")
                output_img_base64 = result.get("output_image_base64", None)

                saved_paths = []
                if output_img_base64:
                    img_bytes = base64.b64decode(output_img_base64)
                    local_path = os.path.join(output_dir, f"{file_prefix}_{uuid.uuid4().hex}_0.jpg")
                    with open(local_path, "wb") as f:
                        f.write(img_bytes)
                    saved_paths.append(local_path)

                return saved_paths

            else:
                image = kwargs.get("input_image")
                print("aaaaaaaaaaaaaaaaaaaaaaaaaaaa-------",image)
                
                # 发起API请求
                response = await self._make_request(model_name, prompt, **kwargs)
                
                if model_name=="gemini-2.5-flash-image-preview":
                    try:
                        # response 直接是 base64 字符串
                        img_data = base64.b64decode(response)

                        # 保存图片
                        os.makedirs(output_dir, exist_ok=True)
                        file_name = f"{file_prefix}_0.png"
                        file_path = os.path.join(output_dir, file_name)

                        with open(file_path, "wb") as f:
                            f.write(img_data)

                        print(f"✅ Image saved to: {file_path}")
                        
                        saved_paths = []
                        # 构造 downloaded_paths 列表，保持后续逻辑一致
                        saved_paths.append(file_path)

                    except Exception as e:
                        print(f"❌ 保存 base64 图片失败: {e}")
                
                else:

                    # 提取图片数据
                    images = self._extract_images_from_response(response, model_name)
                    
                    # --- START: 关键的调试代码 ---
                    if not images:
                        logger.warning("❌ 响应中未找到可供保存的图片数据。")
                        # 如果找不到图片，并且处于调试模式，就打印完整的原始响应
                        if self.debug:
                            print("\n" + "="*25 + " DEBUG: RAW RESPONSE " + "="*25)
                            print(f"未能从 {model_name} 的响应中提取图片，以下是原始数据：")
                            print(json.dumps(response, indent=2, ensure_ascii=False))
                            print("="*70 + "\n")
                        return []
                    # --- END: 关键的调试代码 ---

                    # 保存图片
                    saved_paths = []
                    download_tasks = []

                    for i, image in enumerate(images):
                        try:
                            if self.debug:
                                print(f"\n💾 准备保存第{i+1}张图片 ({image['type']})...")
                            
                            if image["type"] == "base64":
                                ext = image.get("mimeType", "png")
                                if ext not in ["png", "jpg", "jpeg", "webp", "gif"]:
                                    ext = "png"
                                
                                path = self._save_image_from_base64_with_ext(
                                    image["data"], output_dir, file_prefix, i, ext
                                )
                                saved_paths.append(path)
                            elif image["type"] == "url":
                                # 并发下载
                                task = asyncio.create_task(self._download_image_from_url(
                                    image["data"], output_dir, file_prefix, i
                                ))
                                download_tasks.append(task)
                            else:
                                logger.warning(f"未知的图片类型: {image['type']}")
                                continue
                            
                        except Exception as e:
                            logger.error(f"处理第{i+1}张图片时失败: {e}")
                    
                    # 等待所有下载任务完成
                    if download_tasks:
                        downloaded_paths = await asyncio.gather(*download_tasks)
                        saved_paths.extend([p for p in downloaded_paths if p])
                
                return saved_paths

        except Exception as e:
            logger.error(f"生成图片过程失败: {e}")
            # 在调试模式下打印堆栈跟踪，以便更详细地排查问题
            if self.debug:
                import traceback
                traceback.print_exc()
            raise
    
    def _save_image_from_base64_with_ext(self, b64_data: str, output_dir: str, file_prefix: str, index: int = 0, ext: str = "png") -> str:
        """从base64数据保存图片（支持指定扩展名）"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理文件名冲突
        current_index = index
        while True:
            file_name = f"{file_prefix}_{current_index}.{ext}"
            file_path = os.path.join(output_dir, file_name)
            
            if not os.path.exists(file_path):
                break
            current_index += 1
        
        if not isinstance(b64_data, str):
            raise ValueError(f"base64数据必须是字符串，但收到: {type(b64_data)}")
        
        try:
            image_bytes = base64.b64decode(b64_data)
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            
            logger.info(f"图片已保存至: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"base64解码失败: {e}")
            logger.error(f"数据内容预览: {str(b64_data)[:100]}...")
            raise
    
    
    async def generate_images_batch(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        
        # for i, req in enumerate(requests):
        #     print(f"请求 {i+1}:")
        #     print("  model_name:", req.get("model_name"))
        #     print("  prompt:", req.get("prompt"))
        #     print("  input_image:", req.get("input_image"))
        #     print("-" * 40)
        
        
        
        """批量生成图片（带临时文件夹防竞争机制）"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # 创建主临时目录
        temp_base_dir = os.path.join(os.getcwd(), f"temp_batch_{int(time.time() * 1000)}")
        os.makedirs(temp_base_dir, exist_ok=True)
        
        async def generate_single(request_data: Dict[str, Any], request_index: int) -> Dict[str, Any]:
            async with semaphore:
                # 为每个请求创建唯一的临时文件夹
                temp_id = f"{request_index}_{str(uuid.uuid4())[:8]}"
                temp_dir = os.path.join(temp_base_dir, f"temp_{temp_id}")
                
                try:
                    start_time = time.time()
                    print("gdgdgdgdgdgdgdgddg",request_data['input_image'])
                    # 修改请求数据，使用临时输出目录
                    temp_request = request_data.copy()

                    # print("aaaaaaaaaaa prompt:", temp_request.get("prompt"))

                    original_output_dir = temp_request.get('output_dir', '.')
                    temp_request['output_dir'] = temp_dir
                    
                    # 生成图片到临时目录
                    temp_paths = await self.generate_image(**temp_request)
                    
                    # 移动文件到最终目录
                    final_paths = []
                    os.makedirs(original_output_dir, exist_ok=True)
                    
                    for temp_path in temp_paths:
                        if os.path.exists(temp_path):
                            filename = os.path.basename(temp_path)
                            
                            # 处理目标文件名冲突
                            final_path = os.path.join(original_output_dir, filename)
                            counter = 0
                            base_name, ext = os.path.splitext(filename)
                            while os.path.exists(final_path):
                                counter += 1
                                new_filename = f"{base_name}_{counter}{ext}"
                                final_path = os.path.join(original_output_dir, new_filename)
                            
                            # 移动文件
                            shutil.move(temp_path, final_path)
                            final_paths.append(final_path)
                            logger.info(f"文件已移动: {temp_path} -> {final_path}")
                    
                    end_time = time.time()
                    
                    return {
                        "request": request_data,
                        "success": True,
                        "paths": final_paths,
                        "duration": end_time - start_time,
                        "error": None
                    }
                    
                except Exception as e:
                    logger.error(f"批量生成请求 {request_index} 失败: {e}")
                    return {
                        "request": request_data,
                        "success": False,
                        "paths": [],
                        "duration": 0,
                        "error": str(e)
                    }
                finally:
                    # 清理单个请求的临时目录
                    if os.path.exists(temp_dir):
                        try:
                            shutil.rmtree(temp_dir)
                            logger.debug(f"已清理临时目录: {temp_dir}")
                        except Exception as e:
                            logger.warning(f"清理临时目录失败: {temp_dir} - {e}")
        
        try:
            # 并发执行所有请求
            tasks = [generate_single(req, i) for i, req in enumerate(requests)]
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
        
        finally:
            # 清理主临时目录
            if os.path.exists(temp_base_dir):
                try:
                    shutil.rmtree(temp_base_dir)
                    logger.info(f"已清理主临时目录: {temp_base_dir}")
                except Exception as e:
                    logger.warning(f"清理主临时目录失败: {temp_base_dir} - {e}")
    
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

async def test_batch_generation():
    """测试批量图片生成功能"""
    print("🧪 开始测试批量图片生成功能...")
    print("=" * 60)
    
    async with ImageGenerationAPI(debug=False) as api:
        
        # 测试案例1: 基本批量测试
        print("\n📋 测试案例1: 基本批量测试")
        print("-" * 40)
        
        basic_requests = [
            {
                "model_name": "gpt-image-1",
                "prompt": "A red apple on a wooden table",
                "output_dir": "./test_batch_output",
                "file_prefix": "apple",
                "num_images": 1
            },
            {
                "model_name": "gpt-image-1", 
                "prompt": "A blue ocean with white waves",
                "output_dir": "./test_batch_output",
                "file_prefix": "ocean",
                "num_images": 1
            },
            {
                "model_name": "gpt-image-1",
                "prompt": "A green forest in spring",
                "output_dir": "./test_batch_output", 
                "file_prefix": "forest",
                "num_images": 1
            }
        ]
        
        start_time = time.time()
        results = await api.generate_images_batch(basic_requests, max_concurrent=3)
        end_time = time.time()
        
        print(f"⏱️  总耗时: {end_time - start_time:.2f}秒")
        print(f"📊 结果统计:")
        successful = sum(1 for r in results if r['success'])
        print(f"   ✅ 成功: {successful}/{len(results)}")
        print(f"   ❌ 失败: {len(results) - successful}/{len(results)}")
        
        for i, result in enumerate(results):
            if result['success']:
                print(f"   请求{i+1}: ✅ 生成了 {len(result['paths'])} 张图片")
                for path in result['paths']:
                    print(f"      📁 {path}")
            else:
                print(f"   请求{i+1}: ❌ {result['error']}")
        
        # 测试案例2: 高并发相同输出目录测试（测试竞争条件）
        print("\n📋 测试案例2: 高并发相同输出目录测试")
        print("-" * 40)
        
        # 创建多个相似的请求，模拟竞争条件
        concurrent_requests = []
        for i in range(8):
            concurrent_requests.append({
                "model_name": "gpt-image-1",
                "prompt": f"A cute cat number {i+1}",
                "output_dir": "./test_concurrent_output",  # 相同输出目录
                "file_prefix": "cat",  # 相同前缀，容易产生文件名冲突
                "num_images": 1
            })
        
        start_time = time.time()
        concurrent_results = await api.generate_images_batch(concurrent_requests, max_concurrent=6)
        end_time = time.time()
        
        print(f"⏱️  总耗时: {end_time - start_time:.2f}秒")
        print(f"📊 结果统计:")
        successful = sum(1 for r in concurrent_results if r['success'])
        print(f"   ✅ 成功: {successful}/{len(concurrent_results)}")
        print(f"   ❌ 失败: {len(concurrent_results) - successful}/{len(concurrent_results)}")
        
        # 检查文件名是否有冲突
        all_files = []
        for result in concurrent_results:
            if result['success']:
                all_files.extend(result['paths'])
        
        unique_files = set(all_files)
        if len(all_files) == len(unique_files):
            print("   🎯 文件名冲突处理: ✅ 无重复文件名")
        else:
            print("   ⚠️  文件名冲突处理: ❌ 发现重复文件名")
        
        print(f"   📁 生成的文件:")
        for file_path in sorted(all_files):
            print(f"      {file_path}")
        
        # 测试案例3: 混合模型测试
        print("\n📋 测试案例3: 混合模型测试")
        print("-" * 40)
        
        mixed_requests = [
            {
                "model_name": "gpt-image-1",
                "prompt": "A modern city skyline",
                "output_dir": "./test_mixed_output",
                "file_prefix": "gpt_city"
            },
            {
                "model_name": "dall-e-3", 
                "prompt": "A peaceful countryside",
                "output_dir": "./test_mixed_output",
                "file_prefix": "dalle_countryside"
            },
            {
                "model_name": "qwen-image",
                "prompt": "一座古老的中式建筑",
                "output_dir": "./test_mixed_output", 
                "file_prefix": "qwen_building"
            }
        ]
        
        start_time = time.time()
        mixed_results = await api.generate_images_batch(mixed_requests, max_concurrent=3)
        end_time = time.time()
        
        print(f"⏱️  总耗时: {end_time - start_time:.2f}秒")
        for i, result in enumerate(mixed_results):
            model_name = mixed_requests[i]['model_name']
            if result['success']:
                print(f"   {model_name}: ✅ {result['duration']:.2f}秒")
            else:
                print(f"   {model_name}: ❌ {result['error']}")
        
        # 测试案例4: 压力测试（大量并发）
        print("\n📋 测试案例4: 压力测试")
        print("-" * 40)
        
        stress_requests = []
        for i in range(15):  # 15个并发请求
            stress_requests.append({
                "model_name": "gpt-image-1",
                "prompt": f"Test image {i+1}: A simple geometric shape",
                "output_dir": "./test_stress_output",
                "file_prefix": "stress_test",
                "num_images": 1
            })
        
        start_time = time.time()
        stress_results = await api.generate_images_batch(stress_requests, max_concurrent=10)
        end_time = time.time()
        
        print(f"⏱️  总耗时: {end_time - start_time:.2f}秒")
        successful = sum(1 for r in stress_results if r['success'])
        print(f"📊 压力测试结果: {successful}/{len(stress_results)} 成功")
        
        if successful > 0:
            avg_time = sum(r['duration'] for r in stress_results if r['success']) / successful
            print(f"   📈 平均单个请求耗时: {avg_time:.2f}秒")
        
        # 测试案例5: 错误处理测试
        print("\n📋 测试案例5: 错误处理测试")
        print("-" * 40)
        
        error_requests = [
            {
                "model_name": "non-existent-model",  # 不存在的模型
                "prompt": "This should fail",
                "output_dir": "./test_error_output",
                "file_prefix": "error_test"
            },
            {
                "model_name": "gpt-image-1",
                "prompt": "",  # 空提示词
                "output_dir": "./test_error_output",
                "file_prefix": "empty_prompt"
            },
            {
                "model_name": "gpt-image-1",
                "prompt": "A normal request",  # 正常请求
                "output_dir": "./test_error_output",
                "file_prefix": "normal"
            }
        ]
        
        error_results = await api.generate_images_batch(error_requests, max_concurrent=3)
        
        print("📊 错误处理结果:")
        for i, result in enumerate(error_results):
            status = "✅" if result['success'] else "❌"
            print(f"   请求{i+1}: {status}")
            if not result['success']:
                print(f"      错误: {result['error']}")
        
        # 总结
        print("\n" + "=" * 60)
        print("🎯 批量测试完成!")
        print(f"📁 输出目录:")
        print(f"   - ./test_batch_output")
        print(f"   - ./test_concurrent_output") 
        print(f"   - ./test_mixed_output")
        print(f"   - ./test_stress_output")
        print(f"   - ./test_error_output")
        print("\n请检查这些目录中的生成图片！")


# 简化版快速测试
async def quick_batch_test():
    """快速批量测试"""
    print("🚀 快速批量测试...")
    
    async with ImageGenerationAPI(debug=False) as api:
        requests = [
            {
                "model_name": "gpt-image-1",
                "prompt": f"A simple test image {i+1}",
                "output_dir": "./quick_test_output",
                "file_prefix": "quick",
                "num_images": 1
            }
            for i in range(5)
        ]
        
        start_time = time.time()
        results = await api.generate_images_batch(requests, max_concurrent=3)
        end_time = time.time()
        
        successful = sum(1 for r in results if r['success'])
        print(f"✅ 完成: {successful}/{len(results)} 成功，耗时 {end_time-start_time:.2f}秒")
        
        return results


# 添加到主函数中
async def main():
    """主函数"""
    print("🚀 图片生成API工具")
    print("=" * 50)
    
    print("选择模式:")
    print("1. 调试特定模型")
    print("2. 完整测试")
    print("3. 快速测试")
    print("4. 批量功能测试")  # 新增
    print("5. 快速批量测试")  # 新增
    
    mode = input("请输入选择 (1/2/3/4/5): ").strip()
    
    if mode == "1":
        model_name = input("输入要调试的模型名 (如: gpt-image-1): ").strip()
        if not model_name:
            model_name = "gpt-image-1"
        await debug_single_model(model_name)
    elif mode == "2":
        await full_test()
    elif mode == "3":
        await debug_single_model("gpt-image-1", "A cute kitten playing with a ball")
    elif mode == "4":
        await test_batch_generation()
    elif mode == "5":
        await quick_batch_test()
    else:
        print("无效选择，默认执行快速测试")
        await debug_single_model("gpt-image-1", "A cute kitten playing with a ball")


if __name__ == "__main__":
    asyncio.run(main())
