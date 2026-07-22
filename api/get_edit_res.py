

'''qwen image edit example
curl https://aihubmix.com/v1/models/qianfan/qwen-image-edit/predictions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $AIHUBMIX_API_KEY" \
    -d '{
  "input": {
    "prompt": "加一个会飞的猪",
    "image": "https://api-bucket.oss-cn-shenzhen.aliyuncs.com/images/s0004_3d_colored.jpg"
  }
}'
'''

# the field mask is required when feature is erase or repaint
'''irag example
curl https://aihubmix.com/v1/models/qianfan/ernie-irag-edit/predictions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $AIHUBMIX_API_KEY" \
    -d '{
  "input": {
    "prompt": "加一个会飞的猪",
    "image": "https://api-bucket.oss-cn-shenzhen.aliyuncs.com/images/s0004_3d_colored.jpg"
  }
}'
'''


# todo:这段代码调用qwen image edit总是出错说connection error，但是curl命令则是正常的。请你帮我排查问题，分享问题的成因并且给我测试实例去证实你的猜测


from .get_url import upload_image_to_oss

import urllib3

import os
import time
import json
import base64
import mimetypes
import asyncio
import requests
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import shutil
import uuid
import yaml
from PIL import Image
from openai import OpenAI, AsyncOpenAI
import hashlib

class ModelType(Enum):
    """支持的模型类型"""
    OPENAI_EDIT = "openai_edit"
    QWEN_EDIT = "qwen_edit"
    GEMINI_IMAGE = "gemini_image"
    DOUBAO_EDIT = "doubao_edit"  
    DALL_EDIT = "dall_edit"
    IMAGEN_EDIT = "imagen_edit"
    MINGUNIVISION_EDIT = "mingunivision_edit"


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    type: ModelType
    endpoint: Optional[str] = None
    default_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.default_params is None:
            self.default_params = {}


class ImageEditAPI:
    """支持多种图片编辑模型的统一API客户端"""
    
    MODEL_CONFIGS = {
        # OpenAI风格的图片编辑模型
        "gpt-image-1-mini": ModelConfig(
            name="gpt-image-1-mini",
            type=ModelType.OPENAI_EDIT,
            default_params={
                "size": "1024x1536",
                # "input_fidelity": "high",
                # "quality": "high",
                "n": 1
            }
        ),
        
        # Qwen风格的图片编辑模型 - 修复endpoint
        "qwen-image-edit": ModelConfig(
            name="qwen-image-edit",
            type=ModelType.QWEN_EDIT,
            endpoint="/models/qianfan/qwen-image-edit/predictions",
            default_params={}
        ),
        
        # # iRAG 图像编辑模型
        # "ernie-irag-edit": ModelConfig(
        #     name="ernie-irag-edit",
        #     type=ModelType.QWEN_EDIT,
        #     endpoint="/models/qianfan/ernie-irag-edit/predictions",
        #     default_params={}
        # ),
        
        # Gemini风格的多模态模型
        "gemini-2.5-flash-image-preview": ModelConfig(
            name="gemini-2.5-flash-image-preview",
            type=ModelType.GEMINI_IMAGE,
            default_params={
                "modalities": ["text", "image"],
                "temperature": 0.7,
                "max_tokens": 16384
            }
        ),

        # Doubao Seedream 文生图模型（AIHubMix）
        "doubao-seedream": ModelConfig(
            name="doubao-seedream-4-0-250828",
            type=ModelType.DOUBAO_EDIT,
            endpoint="/models/doubao/doubao-seedream-4-0-250828/predictions",
            default_params={}
        ),

        "dall-e-3": ModelConfig(
            name="dall-e-3",
            type=ModelType.DALL_EDIT,
            endpoint="/models/opanai/dall-e-3/predictions",  
            default_params={"size": "1024x1024", "n": 1}
        ),

        "imagen-4.0-fast": ModelConfig(
            name="imagen-4.0-fast-generate-001",
            type=ModelType.IMAGEN_EDIT,
            endpoint="/models/google/imagen-4.0-fast-generate-001/predictions",
            default_params={"numberOfImages": 1}
        ),

        "Ming-UniVision_EDIT": ModelConfig(
            name="Ming-UniVision_EDIT",
            type=ModelType.MINGUNIVISION_EDIT
        ),

        "Ming-UniVision_EDIT4GEN": ModelConfig(
            name="Ming-UniVision_EDIT4GEN",
            type=ModelType.MINGUNIVISION_EDIT
        ),

    }

    def __init__(
        self,
        config_path: str = "./api/config.yaml",
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        初始化多模型图片API客户端
        """
        if model_name == 'Ming-UniVision_EDIT' or model_name == 'Ming-UniVision_EDIT4GEN':
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"配置文件不存在: {config_path}")

            with open(config_path, "r", encoding="utf-8") as f:
                self.config: Dict[str, Any] = yaml.safe_load(f)
            
            self.model_name = model_name or self.config.get("model_name") or (_ for _ in ()).throw(ValueError("❌ 未指定模型名称"))

            self.max_retries = int(self.config.get("max_retries", 1))
            self.retry_delay = float(self.config.get("retry_delay", 2))

            # 验证模型是否支持
            if self.model_name not in self.MODEL_CONFIGS:
                raise ValueError(f"不支持的模型: {self.model_name}，支持的模型: {list(self.MODEL_CONFIGS.keys())}")

            self.model_config = self.MODEL_CONFIGS[self.model_name]
            print(f"[ImageEditAPI] 初始化完成，模型: {self.model_name} ")
            
        else:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"配置文件不存在: {config_path}")

            with open(config_path, "r", encoding="utf-8") as f:
                self.config: Dict[str, Any] = yaml.safe_load(f)

            # 配置优先级: 显式参数 > 配置文件 > 默认
            self.api_key = api_key or self.config.get("api_key") or ""
            self.base_url = base_url or self.config.get("base_url") or "https://aihubmix.com/v1"
            self.model_name = model_name or self.config.get("model_name") or (_ for _ in ()).throw(ValueError("❌ 未指定模型名称"))

            self.max_retries = int(self.config.get("max_retries", 1))
            self.retry_delay = float(self.config.get("retry_delay", 2))

            if not self.api_key:
                raise ValueError("缺少 api_key，请在 ./api/config.yaml 中设置 api_key")

            # 验证模型是否支持
            if self.model_name not in self.MODEL_CONFIGS:
                raise ValueError(f"不支持的模型: {self.model_name}，支持的模型: {list(self.MODEL_CONFIGS.keys())}")

            self.model_config = self.MODEL_CONFIGS[self.model_name]

            # 根据模型类型初始化不同的客户端
            if self.model_config.type in [ModelType.OPENAI_EDIT, ModelType.GEMINI_IMAGE, ModelType.IMAGEN_EDIT, ModelType.DALL_EDIT]:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

            print(f"[ImageEditAPI] 初始化完成，模型: {self.model_name} ({self.model_config.type.value})，网关: {self.base_url}")
        

    @staticmethod
    def encode_image(image_path: str) -> str:
        """将本地图片读取并编码为 base64 字符串"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def guess_mime_type(image_path: str) -> str:
        """根据文件后缀猜测 MIME 类型，默认 image/jpeg"""
        mime, _ = mimetypes.guess_type(image_path)
        return mime or "image/jpeg"

    @staticmethod
    def _ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def serialize_usage(usage_obj) -> Dict[str, Any]:
        """安全地序列化usage对象"""
        if not usage_obj:
            return {}
        
        result = {}
        for attr in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
            try:
                value = getattr(usage_obj, attr, None)
                if value is not None:
                    # 如果是复杂对象，尝试转换为基本类型
                    if hasattr(value, '__dict__'):
                        try:
                            result[attr] = value.__dict__
                        except:
                            result[attr] = str(value)
                    else:
                        result[attr] = value
            except Exception as e:
                print(f"[WARN] 无法序列化usage字段 {attr}: {e}")
                result[attr] = None
        
        return result
    
    def _create_temp_dir(self, base_dir: str) -> str:
        """创建临时文件夹"""
        temp_name = f"temp_{uuid.uuid4().hex[:8]}"
        temp_dir = os.path.join(base_dir, temp_name)
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def _safe_move_from_temp(self, temp_file_path: str, target_dir: str, file_prefix: str = "image") -> str:
        """安全地从临时文件夹移动文件到目标目录，避免文件名冲突"""
        self._ensure_dir(target_dir)
        
        # 获取文件扩展名
        _, ext = os.path.splitext(temp_file_path)
        
        # 寻找不冲突的文件名
        current_index = 0
        while True:
            target_filename = f"{file_prefix}_{current_index}{ext}"
            target_path = os.path.join(target_dir, target_filename)
            
            # 使用原子操作检查并移动
            try:
                # 在移动前再次检查目标文件是否存在
                if not os.path.exists(target_path):
                    shutil.move(temp_file_path, target_path)
                    return target_path
            except (OSError, shutil.Error):
                # 如果移动失败（可能是并发冲突），尝试下一个文件名
                pass
            
            current_index += 1
            if current_index > 10000:  # 防止无限循环
                raise RuntimeError("无法找到可用的文件名")

    def _cleanup_temp_dir(self, temp_dir: str):
        """清理临时文件夹"""
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"[WARN] 清理临时文件夹失败 {temp_dir}: {e}")
            
            
    def _save_images_from_openai_response(self, result, output_dir: str = ".", file_prefix: str = "image_edit") -> List[str]:
        """保存OpenAI风格响应中的图片（支持并发安全）"""
        self._ensure_dir(output_dir)
        saved_paths = []
        
        # 创建临时文件夹
        temp_dir = self._create_temp_dir(output_dir)
        
        try:
            for i, image_item in enumerate(result.data):
                image_base64 = image_item.b64_json
                if image_base64 is None:
                    print(f"警告：第 {i+1} 张图片没有返回 base64 数据，跳过保存。")
                    continue

                image_bytes = base64.b64decode(image_base64)

                # 先保存到临时文件夹
                temp_filename = f"temp_image_{i}.png"
                temp_file_path = os.path.join(temp_dir, temp_filename)
                
                try:
                    with open(temp_file_path, "wb") as f:
                        f.write(image_bytes)
                    
                    # 安全移动到最终位置
                    final_path = self._safe_move_from_temp(temp_file_path, output_dir, file_prefix)
                    saved_paths.append(final_path)
                    print(f"第 {i+1} 张编辑后的图片已保存至：{final_path}")
                    
                except Exception as e:
                    print(f"保存第 {i+1} 张图片时出错: {e}")
        
        finally:
            # 清理临时文件夹
            self._cleanup_temp_dir(temp_dir)

        return saved_paths


    def _save_inline_images(self, parts: List[Dict[str, Any]], save_dir: Optional[str] = None) -> List[str]:
        """保存多模态响应中的图片（Gemini风格，支持并发安全）"""
        if not parts or not save_dir:
            return []

        self._ensure_dir(save_dir)
        saved_paths: List[str] = []
        
        # 创建临时文件夹
        temp_dir = self._create_temp_dir(save_dir)
        
        try:
            for idx, part in enumerate(parts):
                if "inline_data" in part and part["inline_data"]:
                    data = part["inline_data"].get("data")
                    mime_type = part["inline_data"].get("mime_type", "image/png")
                    if not data:
                        continue
                        
                    try:
                        img_bytes = base64.b64decode(data)
                        img = Image.open(BytesIO(img_bytes))
                        
                        # 确定文件扩展名
                        ext = "png"
                        if mime_type == "image/jpeg" or mime_type == "image/jpg":
                            ext = "jpg"
                        elif mime_type == "image/webp":
                            ext = "webp"
                        elif mime_type == "image/png":
                            ext = "png"

                        # 先保存到临时文件夹
                        temp_filename = f"temp_gemini_{idx}.{ext}"
                        temp_file_path = os.path.join(temp_dir, temp_filename)
                        img.save(temp_file_path)
                        
                        # 安全移动到最终位置
                        final_path = self._safe_move_from_temp(temp_file_path, save_dir, "gemini_out")
                        saved_paths.append(final_path)
                        
                    except Exception as e:
                        print(f"[WARN] 解析/保存返回图片失败: {e}")
        
        finally:
            # 清理临时文件夹
            self._cleanup_temp_dir(temp_dir)
            
        return saved_paths
    
    def edit_image_openai_style(
        self,
        image_path: str,
        prompt: str,
        save_dir: str = ".",
        **kwargs
    ) -> Dict[str, Any]:
        """OpenAI风格的图片编辑"""
        # 合并默认参数和用户参数
        params = {**self.model_config.default_params, **kwargs}
        
        result = self.client.images.edit(
            model=self.model_name,
            image=open(image_path, "rb"),
            prompt=prompt,
            **params
        )

        # 保存图片
        saved_paths = self._save_images_from_openai_response(result, save_dir)

        return {
            "text": [f"Generated {len(result.data)} images"],
            "image_paths": saved_paths,
            "usage": self.serialize_usage(result.usage) if hasattr(result, 'usage') else {},
            "raw": result
        }


    def edit_image_qwen_style(
        self,
        image_url: str,
        prompt: str,
        save_dir: str = "./output",
        **kwargs
    ) -> Dict[str, Any]:
        """Qwen风格的图片编辑（修复版）"""
        endpoint = self.base_url + self.model_config.endpoint
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        }
        
        payload = {
            "input": {
                "prompt": prompt,
                "image": image_url
            }
        }
        
        # 创建 Session 以复用连接
        session = requests.Session()
        
        # 配置重试策略（兼容新旧版本的 urllib3）
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        try:
            # 尝试使用新版本的参数名
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["POST"]  # 新版本使用 allowed_methods
            )
        except TypeError:
            # 回退到旧版本的参数名
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                method_whitelist=["POST"]  # 旧版本使用 method_whitelist
            )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        try:
            print(f"[{self.model_name}] 发送请求到: {endpoint}")
            print(f"[{self.model_name}] Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            
            # 关键修复：添加超时设置
            response = session.post(
                endpoint, 
                json=payload,
                headers=headers,
                timeout=(10, 240),  # (连接超时, 读取超时)
                verify=True
            )
            
            print(f"[{self.model_name}] 响应状态码: {response.status_code}")
            
            response.raise_for_status()
            
            result = response.json()
            print(f"[{self.model_name}] API 响应: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")

            # 解析返回的图片 URL
            output_list = result.get("output", [])
            if not output_list or not isinstance(output_list, list):
                raise ValueError(f"API 返回格式异常：{result}")

            output_image_url = output_list[0].get("url")
            if not output_image_url:
                raise ValueError(f"API 返回中未找到图片 URL：{result}")

            # 下载图片到本地
            self._ensure_dir(save_dir)
            
            # 确定文件扩展名
            ext = ".jpg"
            if ".png" in output_image_url.lower():
                ext = ".png"
            elif ".jpeg" in output_image_url.lower():
                ext = ".jpeg"
            elif ".webp" in output_image_url.lower():
                ext = ".webp"

            local_filename = f"edited_{uuid.uuid4().hex[:8]}{ext}"
            local_path = os.path.join(save_dir, local_filename)

            # 下载图片
            print(f"[{self.model_name}] 下载图片: {output_image_url}")
            img_resp = session.get(output_image_url, timeout=(10, 60))
            img_resp.raise_for_status()
            
            with open(local_path, "wb") as f:
                f.write(img_resp.content)

            print(f"[{self.model_name}] 图片已保存至: {local_path}")

            return {
                "text": [prompt],
                "image_paths": [local_path],
                "output_image_url": output_image_url,
                "usage": {},
                "raw": result
            }

        except requests.exceptions.Timeout as e:
            error_msg = f"请求超时: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return {
                "text": ["请求超时"],
                "image_paths": [],
                "output_image_url": "",
                "error": error_msg,
                "raw": {}
            }
        
        except requests.exceptions.ConnectionError as e:
            error_msg = f"连接错误: {str(e)}"
            print(f"[ERROR] {error_msg}")
            if hasattr(e, '__cause__'):
                print(f"[ERROR] 原因: {e.__cause__}")
            return {
                "text": ["连接失败"],
                "image_paths": [],
                "output_image_url": "",
                "error": error_msg,
                "raw": {}
            }
        
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP 错误: {str(e)}"
            if e.response is not None:
                try:
                    detail = e.response.text
                    print(f"[ERROR] 服务器响应: {detail}")
                    error_msg += f" | 详情: {detail}"
                except Exception:
                    pass
            print(f"[ERROR] {error_msg}")
            return {
                "text": ["HTTP 错误"],
                "image_paths": [],
                "output_image_url": "",
                "error": error_msg,
                "raw": {}
            }
        
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "text": ["处理失败"],
                "image_paths": [],
                "output_image_url": "",
                "error": error_msg,
                "raw": {}
            }
        
        finally:
            session.close()


    def edit_image_gemini_style(
        self,
        image_path: str,
        prompt: str,
        save_dir: Optional[str] = None,
        return_base64_images: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Gemini风格的多模态处理"""
        # 合并默认参数和用户参数
        params = {**self.model_config.default_params, **kwargs}
        modalities = params.pop("modalities", ["text", "image"])
        temperature = params.pop("temperature", 0.7)
        max_tokens = params.pop("max_tokens", 16384)

        base64_img = self.encode_image(image_path)
        mime_type = self.guess_mime_type(image_path)
        img_data_url = f"data:{mime_type};base64,{base64_img}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_data_url}},
                ],
            }
        ]

        retries = 0
        last_err = None

        while retries < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    modalities=modalities,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **params
                )

                return self._parse_gemini_response(response, save_dir, return_base64_images)

            except Exception as e:
                retries += 1
                last_err = e
                print(f"[ERROR] Gemini调用失败({retries}/{self.max_retries}): {e}")
                if retries >= self.max_retries:
                    break
                time.sleep(self.retry_delay)

        raise RuntimeError(f"调用 Gemini 模型失败: {last_err}")

    def edit_image_doubao_style(
        self,
        prompt: str,
        save_dir: str = "./output",
        **kwargs
    ) -> Dict[str, Any]:
        """Doubao Seedream 文生图接口"""
        endpoint = self.base_url + self.model_config.endpoint
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "input": {
                "prompt": prompt,
                "size": kwargs.get("size", "2K"),
                "sequential_image_generation": "disabled",
                "stream": False,
                "response_format": "url",
                "watermark": False
            }
        }

        session = requests.Session()
        saved_paths = []
        os.makedirs(save_dir, exist_ok=True)

        try:
            response = session.post(endpoint, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()

            output_list = result.get("output", [])
            for i, item in enumerate(output_list):
                url = item.get("url")
                if not url:
                    continue
                img_bytes = requests.get(url, timeout=60).content
                local_path = os.path.join(save_dir, f"doubao_gen_{i}.jpg")
                with open(local_path, "wb") as f:
                    f.write(img_bytes)
                saved_paths.append(local_path)

            return {
                "text": [prompt],
                "image_paths": saved_paths,
                "raw": result
            }

        except Exception as e:
            error_msg = f"调用 Doubao Seedream 失败: {e}"
            print(f"[ERROR] {error_msg}")
            return {
                "text": ["处理失败"],
                "image_paths": [],
                "error": error_msg,
                "raw": {}
            }
        finally:
            session.close()

    def edit_image_dall_style(
        self,
        prompt: str,
        save_dir: str = "./output",
        **kwargs
    ) -> Dict[str, Any]:
        """DALL·E-3 文生图接口（修复版）"""
        endpoint = self.base_url + self.model_config.endpoint
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        prompt = "请生成一只可爱的小猫。"
        print( f"prompt: {prompt} ")

        # ✅ 关键修改：参数合并逻辑
        params = {**self.model_config.default_params, **kwargs}
        params["prompt"] = prompt
        
        # ✅ 关键修改：标准的 AIHubMix 请求格式
        payload = {"input": params}

        payload = {
            "input": {
                "prompt": prompt,
                "size": "1024x1024", 
                "n": 1
            }
        }

        os.makedirs(save_dir, exist_ok=True)
        session = requests.Session()
        saved_paths = []

        try:
            # ✅ 新增：调试日志
            print(f"[{self.model_name}] 请求URL: {endpoint}")
            print(f"[{self.model_name}] 请求载荷: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            
            response = session.post(endpoint, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()

            # ✅ 新增：响应日志
            print(f"[{self.model_name}] 响应: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")

            output_list = result.get("output", [])
            if not output_list:
                raise ValueError(f"API返回中没有图片数据: {result}")

            for i, item in enumerate(output_list):
                url = item.get("url")
                if not url:
                    print(f"[WARN] 第 {i+1} 个输出项没有URL，跳过")
                    continue
                
                img_resp = session.get(url, timeout=60)
                img_resp.raise_for_status()
                
                # ✅ 改进：更智能的扩展名检测
                ext = os.path.splitext(url.split('?')[0])[1] or ".png"
                local_path = os.path.join(save_dir, f"dalle3_gen_{i}{ext}")
                
                with open(local_path, "wb") as f:
                    f.write(img_resp.content)
                
                saved_paths.append(local_path)
                print(f"[{self.model_name}] 图片已保存: {local_path}")

            return {
                "text": [prompt],
                "image_paths": saved_paths,
                "raw": result
            }

        except requests.exceptions.HTTPError as e:
            # ✅ 增强：更详细的错误处理
            error_msg = f"HTTP错误: {e}"
            if e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"[ERROR] 服务器返回: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                    error_msg += f" | 详情: {error_detail}"
                except:
                    error_msg += f" | 原始响应: {e.response.text}"
            
            print(f"[ERROR] {error_msg}")
            return {
                "text": ["处理失败"],
                "image_paths": [],
                "error": error_msg,
                "raw": {}
            }
        
        except Exception as e:
            error_msg = f"DALL·E-3 生成失败: {e}"
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "text": ["处理失败"],
                "image_paths": [],
                "error": error_msg,
                "raw": {}
            }
        finally:
            session.close()


    def edit_image_imagen_style(
        self,
        prompt: str,
        save_dir: str = "./output",
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用 Imagen 4.0 Fast 生成图像（文生图），不使用输入图片
        """
        import os, json, requests

        endpoint = self.base_url + self.model_config.endpoint
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        prompt = "请生成一只可爱的小猫。"
        print( f"prompt: {prompt} ")

        # 构造 payload
        # ✅ 关键点：prompt 在 input 内，numberOfImages 默认 1
        payload = {
            "input": {
                "prompt": prompt,
                "numberOfImages": kwargs.get("numberOfImages", 1)
            }
        }

        # 其他 kwargs 字段直接加入 input，避免覆盖 prompt 和 numberOfImages
        # for k, v in kwargs.items():
        #     if k not in ["prompt", "numberOfImages", "image"]:
        #         payload["input"][k] = v

        os.makedirs(save_dir, exist_ok=True)
        session = requests.Session()
        saved_paths = []

        try:
            print(f"[{self.model_name}] 请求URL: {endpoint}")
            print(f"[{self.model_name}] 请求载荷: {json.dumps(payload, indent=2, ensure_ascii=False)}")

            response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()

            print(f"[{self.model_name}] 响应: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")

            output_list = result.get("output", [])
            if not output_list:
                raise ValueError(f"API返回中没有图片数据: {result}")

            for i, item in enumerate(output_list):
                url = item.get("url")
                if not url:
                    print(f"[WARN] 第 {i+1} 个输出项没有URL，跳过")
                    continue

                img_resp = session.get(url, timeout=60)
                img_resp.raise_for_status()

                ext = os.path.splitext(url.split('?')[0])[1] or ".jpg"
                local_path = os.path.join(save_dir, f"imagen_gen_{i}{ext}")

                with open(local_path, "wb") as f:
                    f.write(img_resp.content)

                saved_paths.append(local_path)
                print(f"[{self.model_name}] 图片已保存: {local_path}")

            return {
                "text": [prompt],
                "image_paths": saved_paths,
                "raw": result
            }

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP错误: {e}"
            if e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"[ERROR] 服务器返回: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                    error_msg += f" | 详情: {error_detail}"
                except:
                    error_msg += f" | 原始响应: {e.response.text}"
            print(f"[ERROR] {error_msg}")
            return {
                "text": ["处理失败"],
                "image_paths": [],
                "error": error_msg,
                "raw": {}
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"Imagen 生成失败: {e}"
            print(f"[ERROR] {error_msg}")
            return {
                "text": ["处理失败"],
                "image_paths": [],
                "error": error_msg,
                "raw": {}
            }

        finally:
            session.close()


    def edit_image_mingunivision_style(
            self,
            input_image_path: str,
            prompt: str,
            save_dir: str = "./output",
            **kwargs
    ) -> Dict[str, Any]:
        """Ming-UniVision 文生图接口（与 Doubao 命名逻辑对齐）"""
        import requests, base64, os

        local_api = "http://127.0.0.1:10091/api/run_model"
        os.makedirs(save_dir, exist_ok=True)

        # === 图片路径转 base64 ===
        input_image_base64 = None
        if input_image_path:
            with open(input_image_path, "rb") as f:
                input_image_base64 = base64.b64encode(f.read()).decode("utf-8")

        # === payload 作为 form-data 发送 ===
        
        if self.model_name == "Ming-UniVision_EDIT":
            payload = {
                "model_name": "Ming-UniVision_EDIT",
                "weight_version": "Ming-UniVision-16B-A3B",
                "mode": "single_editing",
                "input_text": prompt,
                "input_image_base64": input_image_base64
            }
        elif self.model_name == "Ming-UniVision_EDIT4GEN":
            payload = {
                "model_name": "Ming-UniVision_EDIT4GEN",
                "weight_version": "Ming-UniVision-16B-A3B",
                "mode": "single_editing",
                "input_text": prompt,
                "input_image_base64": input_image_base64
            }
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")

        print({
            "model_name": payload["model_name"],
            "weight_version": payload["weight_version"],
            "mode": payload["mode"],
            "input_text": payload["input_text"],
            "input_image_base64_length": len(payload["input_image_base64"]) if payload["input_image_base64"] else 0,
            "input_image_base64_preview": (payload["input_image_base64"][:100] + "...") if payload["input_image_base64"] else None
        })

        try:
            # === 改成 data= 发送 form-data ===
            response = requests.post(local_api, data=payload, timeout=180)
            response.raise_for_status()
            result = response.json()

            # === 拿出 Ming 返回的关键字段 ===
            output_text = result.get("output_text", "")
            output_img_base64 = result.get("output_image_base64", None)

            saved_paths = []
            if output_img_base64:
                img_bytes = base64.b64decode(output_img_base64)
                file_suffix = ".png"
                if self.model_name == "Ming-UniVision_EDIT":
                    file_name = f"edited_{uuid.uuid4().hex[:8]}{file_suffix}"
                elif self.model_name == "Ming-UniVision_EDIT4GEN":
                    file_name = f"generated_{uuid.uuid4().hex[:8]}{file_suffix}"
                local_path = os.path.join(save_dir, file_name)
                with open(local_path, "wb") as f:
                    f.write(img_bytes)
                saved_paths.append(local_path)

            # === 对齐 Doubao 的输出结构 ===
            return {
                "text": [output_text if output_text else prompt],
                "image_paths": saved_paths,
                "raw": result
            }

        except Exception as e:
            error_msg = f"调用 Ming-UniVision 失败: {e}"
            print(f"[ERROR] {error_msg}")
            return {
                "text": ["处理失败"],
                "image_paths": [],
                "error": error_msg,
                "raw": {}
            }


    def _parse_gemini_response(self, response, save_dir: Optional[str] = None, return_base64_images: bool = False) -> Dict[str, Any]:
        """解析Gemini响应"""
        texts: List[str] = []
        image_paths: List[str] = []
        image_base64_list: List[str] = []

        msg = response.choices[0].message

        # 从 multi_mod_content 中解析
        if hasattr(msg, "multi_mod_content") and msg.multi_mod_content:
            for part in msg.multi_mod_content:
                if "text" in part and part["text"]:
                    texts.append(part["text"])
                elif "inline_data" in part and part["inline_data"]:
                    if save_dir:
                        saved = self._save_inline_images([part], save_dir=save_dir)
                        image_paths.extend(saved)
                    elif return_base64_images:
                        data_b64 = part["inline_data"].get("data")
                        if data_b64:
                            image_base64_list.append(data_b64)

        # 备用：从 content 解析纯文本
        if not texts and getattr(msg, "content", None):
            if isinstance(msg.content, str):
                texts.append(msg.content.strip())

        return {
            "text": texts,
            "image_paths": image_paths,
            "image_base64": image_base64_list if return_base64_images else [],
            "usage": self.serialize_usage(response.usage) if hasattr(response, "usage") else {},
            "raw": response,
        }


    def edit_image(
        self,
        image_path: str,
        prompt: str,
        save_dir: str = "./output",
        **kwargs
    ) -> Dict[str, Any]:
        """根据模型类型路由到对应的编辑方法"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        print(f"[{self.model_name}] 开始处理图片编辑...")
        print(f"图片路径: {image_path}")
        print(f"提示词: {prompt}")
        print(f"MODEL_TYPE 类型: {self.model_config.type}")

        if self.model_config.type == ModelType.OPENAI_EDIT:
            return self.edit_image_openai_style(image_path, prompt, save_dir, **kwargs)
        elif self.model_config.type == ModelType.QWEN_EDIT:
            try:
                print(f"[{self.model_name}] 正在上传图片到 OSS...")
                input_image_url = upload_image_to_oss(image_path)
                print(f"[{self.model_name}] 图片已上传，URL: {input_image_url}")
            except Exception as e:
                raise RuntimeError(f"上传图片到 OSS 失败: {e}")
            # 传入 save_dir
            return self.edit_image_qwen_style(
                image_url=input_image_url,
                prompt=prompt,
                save_dir=save_dir,
                **kwargs
            )
        elif self.model_config.type == ModelType.GEMINI_IMAGE:
            return self.edit_image_gemini_style(image_path, prompt, save_dir, **kwargs)
        
        elif self.model_config.type == ModelType.DOUBAO_EDIT:
            return self.edit_image_doubao_style(
                prompt=prompt,
                save_dir=save_dir,
                **kwargs
            )  
        
        elif self.model_config.type == ModelType.DALL_EDIT:
            return self.edit_image_dall_style(
                prompt=prompt,
                save_dir=save_dir,
                size=kwargs.get("size", "1024x1024"),
                n=kwargs.get("n", 1),
                **kwargs
            )

        elif self.model_config.type == ModelType.IMAGEN_EDIT:
            return self.edit_image_imagen_style(
                prompt=prompt,
                save_dir=save_dir,
                **kwargs
            )
     
        elif self.model_config.type == ModelType.MINGUNIVISION_EDIT:
            return self.edit_image_mingunivision_style(
                input_image_path=image_path,
                prompt=prompt,
                save_dir=save_dir,
                **kwargs
            )

        else:
            raise ValueError(f"不支持的模型类型: {self.model_config.type}")

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
                print("tttttttttttttttttt", task_dir)
                if self.model_name == "Ming-UniVision_EDIT" or self.model_name == "Ming-UniVision_EDIT4GEN":
                    print("gggggggg", save_dir)
                    result = self.edit_image(image_path, prompt, save_dir, **kwargs)
                else:
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


    @staticmethod
    def generate_sha_hash(content: bytes) -> str:
        """生成SHA哈希值（取前16位）"""
        return hashlib.sha256(content).hexdigest()[:16]

    @staticmethod
    def rename_generated_images_with_hash(image_paths: List[str], output_dir: str) -> List[str]:
        """将生成的图片重命名为哈希值并移到 output_dir"""
        new_paths = []
        os.makedirs(output_dir, exist_ok=True)

        for original_path in image_paths:
            if not os.path.exists(original_path):
                # 文件可能已被其它并发任务移动/覆盖
                continue

            with open(original_path, 'rb') as f:
                img_content = f.read()

            hash_name = ImageEditAPI.generate_sha_hash(img_content)
            # ext = Path(original_path).suffix.lower() or '.jpg'
            ext = '.png'
            new_path = os.path.join(output_dir, f"{hash_name}{ext}")

            # 防止哈希碰撞导致覆盖
            if os.path.exists(new_path):
                base = f"{hash_name}"
                k = 1
                while os.path.exists(new_path):
                    new_path = os.path.join(output_dir, f"{base}_{k}{ext}")
                    k += 1

            shutil.move(original_path, new_path)
            new_paths.append(new_path)

        return new_paths

    
    async def edit_images_batch_async(
        self,
        image_prompts: List[Tuple[str, str]],
        save_dir: str = "./output",
        max_concurrent: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """异步批量处理图片编辑，并将生成的图片直接保存到 save_dir 根目录（哈希命名）"""
        self._ensure_dir(save_dir)
        
        async def process_single(i: int, image_path: str, prompt: str):
            # 创建临时工作目录（用于 API 内部保存原始输出）
            task_dir = os.path.join(save_dir, f"task_{i:04d}")
            os.makedirs(task_dir, exist_ok=True)
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.edit_image(image_path, prompt, task_dir, **kwargs)
                )
                result["task_index"] = i
                result["image_path"] = image_path
                result["prompt"] = prompt

                # 如果有生成图片，将其从 task_dir 移出并哈希重命名到 save_dir 根目录
                if result.get("image_paths"):
                    new_paths = self.rename_generated_images_with_hash(result["image_paths"], save_dir)
                    result["image_paths"] = new_paths  # 更新为最终路径

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
            finally:
                # 清理临时 task_dir（即使出错也尝试删除）
                try:
                    shutil.rmtree(task_dir, ignore_errors=True)
                except Exception:
                    pass
        
        print(f"[{self.model_name}] 开始异步批量处理 {len(image_prompts)} 个任务...")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_process(i, image_path, prompt):
            async with semaphore:
                return await process_single(i, image_path, prompt)
        
        tasks = [
            bounded_process(i, image_path, prompt)
            for i, (image_path, prompt) in enumerate(image_prompts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
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

    @classmethod
    def get_supported_models(cls) -> List[str]:
        """获取支持的模型列表"""
        return list(cls.MODEL_CONFIGS.keys())

    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[ModelConfig]:
        """获取模型信息"""
        return cls.MODEL_CONFIGS.get(model_name)

    def test_model(self, test_image_path: str, test_prompt: str = "请对这张图片进行优化处理") -> Dict[str, Any]:
        """测试当前模型"""
        try:
            result = self.edit_image(test_image_path, test_prompt)
            return {
                "model": self.model_name,
                "status": "success",
                "result": result
            }
        except Exception as e:
            return {
                "model": self.model_name,
                "status": "error",
                "error": str(e)
            }





# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 测试代码



# def clean_result_for_json(result: Dict[str, Any]) -> Dict[str, Any]:
#     """清理结果中不能JSON序列化的对象"""
#     clean_result = {}
    
#     for key, value in result.items():
#         if key == "raw":
#             # 跳过原始响应对象
#             continue
#         elif key == "usage":
#             # 清理usage对象
#             if hasattr(value, '__dict__'):
#                 clean_result[key] = ImageEditAPI.serialize_usage(value)
#             else:
#                 clean_result[key] = value
#         elif isinstance(value, list):
#             # 处理列表
#             clean_result[key] = [
#                 item.__dict__ if hasattr(item, '__dict__') and not isinstance(item, str) 
#                 else str(item) if not isinstance(item, (str, int, float, bool, type(None)))
#                 else item 
#                 for item in value
#             ]
#         elif hasattr(value, '__dict__') and not isinstance(value, str):
#             # 处理复杂对象
#             try:
#                 clean_result[key] = value.__dict__
#             except:
#                 clean_result[key] = str(value)
#         else:
#             clean_result[key] = value
    
#     return clean_result




# import asyncio
# import time
# import os
# import json
# from typing import List, Tuple, Dict, Any
# import shutil

# def test_batch_functionality(
#     config_path: str = "./api/config.yaml",
#     test_images_dir: str = "./test_images",
#     output_dir: str = "./batch_test_output",
#     model_name: str = "gpt-image-1-mini"
# ) -> Dict[str, Any]:
#     """测试批量处理功能的完整示例"""
    
#     print("🚀 开始批量处理功能测试")
#     print("=" * 60)
    
#     # 清理并创建输出目录
#     if os.path.exists(output_dir):
#         shutil.rmtree(output_dir)
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 1. 准备测试数据
#     test_data = prepare_test_data(test_images_dir)
#     if not test_data["image_prompts"]:
#         print("❌ 没有找到测试图片，请检查test_images目录")
#         return {"error": "No test images found"}
    
#     print(f"📋 准备了 {len(test_data['image_prompts'])} 个测试任务")
#     for i, (img_path, prompt) in enumerate(test_data["image_prompts"]):
#         print(f"   {i+1}. {os.path.basename(img_path)}: {prompt[:50]}...")
    
#     try:
#         # 创建API实例
#         api = ImageEditAPI(config_path=config_path, model_name=model_name)
        
#         # 2. 测试同步批量处理
#         print(f"\n🔄 测试同步批量处理 ({model_name})")
#         print("-" * 40)
#         sync_results = test_sync_batch(api, test_data, output_dir)
        
#         # 3. 测试异步批量处理
#         print(f"\n⚡ 测试异步批量处理 ({model_name})")
#         print("-" * 40)
#         async_results = asyncio.run(test_async_batch(api, test_data, output_dir))
        
#         # 4. 性能对比和结果分析
#         print(f"\n📊 结果分析")
#         print("-" * 40)
#         analysis = analyze_results(sync_results, async_results, test_data)
        
#         # 5. 保存详细报告
#         report = generate_test_report(sync_results, async_results, analysis, test_data)
#         save_test_report(report, output_dir)
        
#         return report
        
#     except Exception as e:
#         print(f"❌ 测试失败: {e}")
#         import traceback
#         traceback.print_exc()
#         return {"error": str(e)}


# def prepare_test_data(test_images_dir: str) -> Dict[str, Any]:
#     """准备测试数据"""
    
#     # 不同类型的处理任务
#     prompts = [
#         "请将这张图片转换为黑白风格",
#         "请对这张图片进行色彩增强",
#         "请为这张图片添加温暖的色调",
#         "请将这张图片转换为卡通风格",
#         "请对这张图片进行锐化处理",
#         "请为这张图片添加复古滤镜",
#         "请将图片亮度提高20%",
#         "请为这张图片添加柔和效果"
#     ]
    
#     image_prompts = []
    
#     # 查找测试图片
#     if os.path.exists(test_images_dir):
#         image_files = [f for f in os.listdir(test_images_dir) 
#                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
#         for i, img_file in enumerate(image_files[:8]):  # 最多取8张图片
#             img_path = os.path.join(test_images_dir, img_file)
#             prompt = prompts[i % len(prompts)]
#             image_prompts.append((img_path, prompt))
    
#     # 如果没有找到图片，创建一些测试用例（需要提供实际的测试图片路径）
#     if not image_prompts:
#         # 这里应该替换为实际存在的图片路径
#         print("⚠️  未找到测试图片目录，使用默认测试图片")
#         test_image = "/data2/wangchangmiao/yjj/medical_AI/medical-bench/data/M3D-RefSeg/images/s0034_3d_colored.jpg"
#         if os.path.exists(test_image):
#             for i in range(min(4, len(prompts))):
#                 image_prompts.append((test_image, prompts[i]))
    
#     return {
#         "image_prompts": image_prompts,
#         "total_tasks": len(image_prompts),
#         "prompts_used": list(set([p for _, p in image_prompts]))
#     }


# def test_sync_batch(api, test_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
#     """测试同步批量处理"""
#     sync_output_dir = os.path.join(output_dir, "sync_batch")
    
#     start_time = time.time()
    
#     try:
#         results = api.edit_images_batch(
#             image_prompts=test_data["image_prompts"],
#             save_dir=sync_output_dir,
#             max_concurrent=3  # 同步模式下这个参数不生效，但保持接口一致
#         )
        
#         end_time = time.time()
#         duration = end_time - start_time
        
#         # 分析结果
#         success_count = sum(1 for r in results if 'error' not in r)
#         error_count = len(results) - success_count
        
#         print(f"✅ 同步批量处理完成")
#         print(f"   耗时: {duration:.2f}秒")
#         print(f"   成功: {success_count}/{len(results)}")
#         print(f"   失败: {error_count}")
        
#         if error_count > 0:
#             print(f"   错误详情:")
#             for i, result in enumerate(results):
#                 if 'error' in result:
#                     print(f"     任务{i+1}: {result['error']}")
        
#         return {
#             "method": "sync",
#             "results": results,
#             "duration": duration,
#             "success_count": success_count,
#             "error_count": error_count,
#             "output_dir": sync_output_dir
#         }
        
#     except Exception as e:
#         print(f"❌ 同步批量处理失败: {e}")
#         return {
#             "method": "sync",
#             "error": str(e),
#             "duration": time.time() - start_time
#         }


# async def test_async_batch(api, test_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
#     """测试异步批量处理"""
#     async_output_dir = os.path.join(output_dir, "async_batch")
    
#     start_time = time.time()
    
#     try:
#         results = await api.edit_images_batch_async(
#             image_prompts=test_data["image_prompts"],
#             save_dir=async_output_dir,
#             max_concurrent=2  # 2为最大并发量
#         )
        
#         end_time = time.time()
#         duration = end_time - start_time
        
#         # 分析结果
#         success_count = sum(1 for r in results if 'error' not in r)
#         error_count = len(results) - success_count
        
#         print(f"✅ 异步批量处理完成")
#         print(f"   耗时: {duration:.2f}秒")
#         print(f"   成功: {success_count}/{len(results)}")
#         print(f"   失败: {error_count}")
#         print(f"   并发数: 3")
        
#         if error_count > 0:
#             print(f"   错误详情:")
#             for i, result in enumerate(results):
#                 if 'error' in result:
#                     print(f"     任务{i+1}: {result['error']}")
        
#         return {
#             "method": "async",
#             "results": results,
#             "duration": duration,
#             "success_count": success_count,
#             "error_count": error_count,
#             "output_dir": async_output_dir
#         }
        
#     except Exception as e:
#         print(f"❌ 异步批量处理失败: {e}")
#         return {
#             "method": "async",
#             "error": str(e),
#             "duration": time.time() - start_time
#         }


# def analyze_results(sync_results: Dict, async_results: Dict, test_data: Dict) -> Dict[str, Any]:
#     """分析测试结果"""
#     analysis = {
#         "total_tasks": test_data["total_tasks"],
#         "performance_comparison": {},
#         "reliability_comparison": {},
#         "output_comparison": {}
#     }
    
#     # 性能对比
#     if "duration" in sync_results and "duration" in async_results:
#         sync_duration = sync_results["duration"]
#         async_duration = async_results["duration"]
#         speedup = sync_duration / async_duration if async_duration > 0 else 0
        
#         analysis["performance_comparison"] = {
#             "sync_duration": sync_duration,
#             "async_duration": async_duration,
#             "speedup_ratio": speedup,
#             "time_saved": sync_duration - async_duration,
#             "faster_method": "async" if async_duration < sync_duration else "sync"
#         }
        
#         print(f"⏱️  性能对比:")
#         print(f"   同步耗时: {sync_duration:.2f}秒")
#         print(f"   异步耗时: {async_duration:.2f}秒")
#         print(f"   加速比: {speedup:.2f}x")
#         print(f"   节省时间: {sync_duration - async_duration:.2f}秒")
    
#     # 可靠性对比
#     sync_success = sync_results.get("success_count", 0)
#     async_success = async_results.get("success_count", 0)
    
#     analysis["reliability_comparison"] = {
#         "sync_success_rate": sync_success / test_data["total_tasks"] if test_data["total_tasks"] > 0 else 0,
#         "async_success_rate": async_success / test_data["total_tasks"] if test_data["total_tasks"] > 0 else 0,
#         "sync_success_count": sync_success,
#         "async_success_count": async_success
#     }
    
#     print(f"🎯 可靠性对比:")
#     print(f"   同步成功率: {sync_success}/{test_data['total_tasks']} ({sync_success/test_data['total_tasks']*100:.1f}%)")
#     print(f"   异步成功率: {async_success}/{test_data['total_tasks']} ({async_success/test_data['total_tasks']*100:.1f}%)")
    
#     return analysis


# def generate_test_report(sync_results: Dict, async_results: Dict, analysis: Dict, test_data: Dict) -> Dict[str, Any]:
#     """生成测试报告"""
    
#     # 清理结果用于JSON序列化
#     clean_sync_results = clean_batch_results(sync_results)
#     clean_async_results = clean_batch_results(async_results)
    
#     report = {
#         "test_info": {
#             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#             "total_tasks": test_data["total_tasks"],
#             "test_prompts": test_data["prompts_used"]
#         },
#         "sync_batch_results": clean_sync_results,
#         "async_batch_results": clean_async_results,
#         "analysis": analysis,
#         "recommendations": generate_recommendations(analysis)
#     }
    
#     return report


# def clean_batch_results(results: Dict[str, Any]) -> Dict[str, Any]:
#     """清理批量处理结果用于JSON序列化"""
#     if "results" in results:
#         cleaned_results = []
#         for result in results["results"]:
#             if isinstance(result, dict):
#                 cleaned_result = clean_result_for_json(result)
#                 cleaned_results.append(cleaned_result)
#             else:
#                 cleaned_results.append(str(result))
        
#         clean_results = {**results}
#         clean_results["results"] = cleaned_results
#         return clean_results
    
#     return results


# def generate_recommendations(analysis: Dict[str, Any]) -> List[str]:
#     """基于分析结果生成建议"""
#     recommendations = []
    
#     perf = analysis.get("performance_comparison", {})
#     reliability = analysis.get("reliability_comparison", {})
    
#     if perf.get("speedup_ratio", 0) > 1.5:
#         recommendations.append("异步批量处理显著快于同步处理，建议在大批量任务时使用异步模式")
#     elif perf.get("speedup_ratio", 0) < 0.8:
#         recommendations.append("同步处理在当前测试中表现更好，可能是因为任务量较小或网络延迟影响")
    
#     sync_success_rate = reliability.get("sync_success_rate", 0)
#     async_success_rate = reliability.get("async_success_rate", 0)
    
#     if abs(sync_success_rate - async_success_rate) < 0.1:
#         recommendations.append("两种方法的成功率相近，可以根据性能需求选择")
#     elif sync_success_rate > async_success_rate:
#         recommendations.append("同步处理成功率更高，在稳定性要求高的场景下建议使用同步模式")
#     else:
#         recommendations.append("异步处理成功率更高，推荐使用异步模式")
    
#     if perf.get("time_saved", 0) > 10:
#         recommendations.append(f"异步处理节省了 {perf['time_saved']:.1f} 秒，在时间敏感的应用中优势明显")
    
#     return recommendations


# def save_test_report(report: Dict[str, Any], output_dir: str):
#     """保存测试报告"""
#     report_path = os.path.join(output_dir, "batch_test_report.json")
    
#     with open(report_path, "w", encoding="utf-8") as f:
#         json.dump(report, f, ensure_ascii=False, indent=2)
    
#     print(f"\n📄 详细测试报告已保存到: {report_path}")
    
#     # 生成简要总结
#     print(f"\n📈 测试总结:")
#     print(f"   总任务数: {report['test_info']['total_tasks']}")
    
#     if "performance_comparison" in report["analysis"]:
#         perf = report["analysis"]["performance_comparison"]
#         print(f"   性能优势: {perf.get('faster_method', 'unknown')} 模式")
#         print(f"   加速比: {perf.get('speedup_ratio', 0):.2f}x")
    
#     print(f"\n💡 建议:")
#     for rec in report["recommendations"]:
#         print(f"   • {rec}")


# # 高级测试功能
# def test_batch_with_different_concurrency(
#     config_path: str = "./api/config.yaml",
#     test_images_dir: str = "./test_images",
#     model_name: str = "gpt-image-1-mini",
#     concurrency_levels: List[int] = [1, 3, 5, 10]
# ) -> Dict[str, Any]:
#     """测试不同并发级别的性能"""
    
#     print("🔬 测试不同并发级别的性能")
#     print("=" * 50)
    
#     test_data = prepare_test_data(test_images_dir)
#     if not test_data["image_prompts"]:
#         return {"error": "No test images found"}
    
#     api = ImageEditAPI(config_path=config_path, model_name=model_name)
#     results = {}
    
#     for max_concurrent in concurrency_levels:
#         print(f"\n🔧 测试并发级别: {max_concurrent}")
#         print("-" * 30)
        
#         output_dir = f"./concurrency_test_{max_concurrent}"
#         start_time = time.time()
        
#         try:
#             async def test_concurrency():
#                 return await api.edit_images_batch_async(
#                     image_prompts=test_data["image_prompts"],
#                     save_dir=output_dir,
#                     max_concurrent=max_concurrent
#                 )
            
#             batch_results = asyncio.run(test_concurrency())
#             duration = time.time() - start_time
#             success_count = sum(1 for r in batch_results if 'error' not in r)
            
#             results[f"concurrent_{max_concurrent}"] = {
#                 "max_concurrent": max_concurrent,
#                 "duration": duration,
#                 "success_count": success_count,
#                 "total_tasks": len(batch_results),
#                 "success_rate": success_count / len(batch_results),
#                 "tasks_per_second": len(batch_results) / duration
#             }
            
#             print(f"   耗时: {duration:.2f}秒")
#             print(f"   成功率: {success_count}/{len(batch_results)} ({success_count/len(batch_results)*100:.1f}%)")
#             print(f"   处理速度: {len(batch_results)/duration:.2f} 任务/秒")
            
#         except Exception as e:
#             results[f"concurrent_{max_concurrent}"] = {
#                 "max_concurrent": max_concurrent,
#                 "error": str(e)
#             }
#             print(f"   ❌ 失败: {e}")
    
#     # 找出最优并发级别
#     valid_results = {k: v for k, v in results.items() if "error" not in v}
#     if valid_results:
#         best_performance = max(valid_results.items(), key=lambda x: x[1]["tasks_per_second"])
#         print(f"\n🏆 最优并发级别: {best_performance[1]['max_concurrent']} (速度: {best_performance[1]['tasks_per_second']:.2f} 任务/秒)")
    
#     return results


# if __name__ == "__main__":
#     # 基础批量处理测试
#     print("开始批量处理功能测试...")
    
#     # 确保有测试图片目录
#     test_images_dir = "./input_image"
#     if not os.path.exists(test_images_dir):
#         os.makedirs(test_images_dir, exist_ok=True)
#         print(f"请在 {test_images_dir} 目录中放入测试图片")
    
#     try:
#         # 基础测试
#         basic_results = test_batch_functionality(
#             test_images_dir=test_images_dir,
#             # model_name="qwen-image-edit"
#             model_name="gpt-image-1-mini"
#         )
        
#         # if "error" not in basic_results:
#         #     print("\n" + "="*60)
#         #     print("🔬 进阶测试: 不同并发级别性能对比")
            
#         #     # 并发级别测试
#         #     concurrency_results = test_batch_with_different_concurrency(
#         #         test_images_dir=test_images_dir,
#         #         concurrency_levels=[1, 5, 10]
#         #     )
            
#         #     # 保存并发测试结果
#         #     with open("./concurrency_test_results.json", "w", encoding="utf-8") as f:
#         #         json.dump(concurrency_results, f, ensure_ascii=False, indent=2)
#         #     print(f"\n并发测试结果已保存到: ./concurrency_test_results.json")
        
#     except Exception as e:
#         print(f"测试失败: {e}")
#         import traceback
#         traceback.print_exc()



def test_qwen_connection():
    """测试 Qwen 图片编辑连接"""
    
    # 1. 基础连接测试
    print("=" * 50)
    print("测试 1: 基础连接测试")
    print("=" * 50)
    
    import requests
    try:
        response = requests.get("https://aihubmix.com", timeout=10)
        print(f"✓ 基础连接成功，状态码: {response.status_code}")
    except Exception as e:
        print(f"✗ 基础连接失败: {e}")
        return
    
    # 2. 测试图片上传
    print("\n" + "=" * 50)
    print("测试 2: 图片上传到 OSS")
    print("=" * 50)
    
    test_image = "/data2/wangchangmiao/yjj/test_AIStation_api/generated_image.png"  # 请替换为你的测试图片路径
    
    if not os.path.exists(test_image):
        print(f"✗ 测试图片不存在: {test_image}")
        return
    
    try:
        image_url = upload_image_to_oss(test_image)
        print(f"✓ 图片上传成功: {image_url}")
    except Exception as e:
        print(f"✗ 图片上传失败: {e}")
        return
    
    # 3. 测试 API 调用（使用 requests 直接测试）
    print("\n" + "=" * 50)
    print("测试 3: 直接 requests 调用 API")
    print("=" * 50)
    
    import yaml
    with open("./api/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    api_key = config.get("api_key")
    base_url = config.get("base_url", "https://aihubmix.com/v1")
    
    endpoint = base_url + "/models/qianfan/qwen-image-edit/predictions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Mozilla/5.0"
    }
    payload = {
        "input": {
            "prompt": "加一个会飞的猪",
            "image": image_url
        }
    }
    
    try:
        print(f"请求 URL: {endpoint}")
        print(f"请求头: {headers}")
        print(f"请求体: {payload}")
        
        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=(10, 240)
        )
        
        print(f"✓ 响应状态码: {response.status_code}")
        print(f"✓ 响应内容: {response.text[:500]}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ API 调用成功")
            print(f"  输出: {result.get('output', [])}")
        else:
            print(f"✗ API 返回错误: {response.text}")
            
    except requests.exceptions.Timeout:
        print("✗ 请求超时")
    except requests.exceptions.ConnectionError as e:
        print(f"✗ 连接错误: {e}")
    except Exception as e:
        print(f"✗ 未知错误: {e}")
    
    # 4. 测试封装的 API 类
    print("\n" + "=" * 50)
    print("测试 4: 使用 ImageEditAPI 类")
    print("=" * 50)
    
    try:
        api = ImageEditAPI(model_name="qwen-image-edit")
        result = api.edit_image(
            image_path=test_image,
            prompt="加一个会飞的猪",
            save_dir="./test_output"
        )
        
        if "error" in result:
            print(f"✗ API 调用失败: {result['error']}")
        else:
            print(f"✓ API 调用成功")
            print(f"  生成图片: {result.get('image_paths', [])}")
            
    except Exception as e:
        print(f"✗ 类调用失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_qwen_connection()
