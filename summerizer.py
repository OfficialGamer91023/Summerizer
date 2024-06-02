import openai
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
import re
import requests
import whisper


openai.api_key = ""

def summarize_text_with_openai(input_text, contenttype):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that produces concise summary."},
            {"role": "user", "content": f"summarize the following {contenttype} concisely. If the text is a tutorial of something, provide the steps too, otherwise only give the summary: {input_text}"},
        ],
        max_tokens=300,
        temperature=1.2,
    )

    return response["choices"][0]["message"]["content"]


def convert_to_text(audio_path : str) -> str:  
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def summarization_interface(input_type, text_input=None, youtube_key=None, article_link=None, audio_file=None):
    if input_type == 'text':
        summary = summarize_text_with_openai(text_input,input_type)
        return summary

    elif input_type == 'youtube video':
        response = YouTubeTranscriptApi.get_transcript(youtube_key)
        transcript = " ".join(entry['text'] for entry in response)
        summary = summarize_text_with_openai(transcript,input_type)
        return summary

    elif input_type == 'article':
        r = requests.get(article_link)
        soup = BeautifulSoup(r.text, 'html.parser')
        content = soup.getText()
        cleaned_content = re.sub(r'\s+', ' ', content).strip()
        summary = summarize_text_with_openai(cleaned_content,input_type)
        return summary

    elif input_type == 'audio':
        transcribed_text = convert_to_text(audio_file)
        summary = summarize_text_with_openai(transcribed_text,input_type)
        return summary
