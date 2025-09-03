import os
from typing import Optional

openai_api_key = "your_openai_key"
claude_api_key = "your_claude_key"
gemini_api_key = "your_gemini_key"

def gpt_call(prompt: str, model: str = "gpt-4o") -> str:
    """OpenAI GPT 모델 호출 함수"""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY") or openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        gpt_client = OpenAI(api_key=api_key)
        res = gpt_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT 호출 오류: {e}")
        raise


def claude_call(prompt: str, model: str = "claude-3-5-sonnet-20241022") -> str:
    """Anthropic Claude 모델 호출 함수"""
    try:
        import anthropic
        api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or claude_api_key
        if not api_key:
            raise ValueError("CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable not set")
            
        claude_client = anthropic.Anthropic(api_key=api_key)
        
        response = claude_client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.content[0].text
    except ImportError:
        print("Error: anthropic 패키지가 설치되지 않았습니다.")
        raise
    except Exception as e:
        print(f"Claude 호출 오류: {e}")
        raise


def gemini_call(prompt: str, model: str = "gemini-2.5-flash") -> str:
    """Google Gemini 모델 호출 함수"""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or gemini_api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
            
        genai.configure(api_key=api_key)
        
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(prompt, generation_config={"temperature": 0.7})
        return response.text
    except ImportError:
        print("Error: google-generativeai 패키지가 설치되지 않았습니다.")
        raise
    except Exception as e:
        print(f"Gemini 호출 오류: {e}")
        raise


def llm_call(prompt: str, model: str = "gpt-4o") -> str:
    """다양한 LLM 모델 호출을 위한 통합 함수"""
    if model.startswith("claude"):
        return claude_call(prompt, model)
    elif model.startswith("gemini"):
        return gemini_call(prompt, model)
    else:  # GPT 모델들
        return gpt_call(prompt, model)


