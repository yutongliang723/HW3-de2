import logging
from botocore.exceptions import ClientError
import boto3
import requests
import isodate
from yake import KeywordExtractor
import googleapiclient.discovery
from youtube_transcript_api import YouTubeTranscriptApi
import re
from transformers import BartTokenizer, BartForConditionalGeneration
import tqdm
import json
import os
import pprint
from typing import Dict
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ComprehendDetect:
    def __init__(self, comprehend_client):
        self.comprehend_client = comprehend_client

    def detect_key_phrases(self, text, language_code):
        try:
            response = self.comprehend_client.detect_key_phrases(
                Text=text, LanguageCode=language_code
            )
            phrases = response["KeyPhrases"]
            logger.info("Detected %s phrases.", len(phrases))
        except ClientError:
            logger.exception("Couldn't detect phrases.")
            raise
        else:
            return phrases
        

    def detect_entities(self, text, language_code):

        try:
            response = self.comprehend_client.detect_entities(
                Text=text, LanguageCode=language_code
            )
            entities = response["Entities"]
            logger.info("Detected %s entities.", len(entities))
        except ClientError:
            logger.exception("Couldn't detect entities.")
            raise
        else:
            return entities
    
    def detect_languages(self, text):

        try:
            response = self.comprehend_client.detect_dominant_language(Text=text)
            languages = response["Languages"]
            logger.info("Detected %s languages.", len(languages))
        except ClientError:
            logger.exception("Couldn't detect languages.")
            raise
        else:
            return languages
def get_keyword_aws(text):
    kw_extractor = KeywordExtractor(lan="en", n=1, dedupLim=0.9, top=10, features=None)
    keywords = kw_extractor.extract_keywords(text)
    keywords_sort = sorted(keywords, key=lambda x: x[-1], reverse=False)[-1]
    return keywords_sort[0]
def get_keyword(article, number):
    text = article['webTitle']
    language_code = 'en'
    comprehend_client = boto3.client(service_name="comprehend", region_name="eu-west-1")
    comprehend_detect = ComprehendDetect(comprehend_client)
    key_phases = comprehend_detect.detect_key_phrases(text, language_code)
    key_phases_sort = sorted(key_phases, key=lambda x: x['Score'], reverse=False)
    key_phases = [get_keyword_aws(key['Text']) for key in key_phases_sort]
    entities = comprehend_detect.detect_entities(text, language_code)
    entities_sort = sorted(entities, key=lambda x: x['Score'],reverse=False)
    entities_found = [e['Text']for e in entities_sort]
    key_words = list(dict.fromkeys(entities_found + key_phases))[:number]
    key_words = " ".join(key_words)
    print("Query: ", key_words)
    return key_words

def get_duration(item):
    video_duration_iso = item['contentDetails']['duration']
    video_duration = isodate.parse_duration(video_duration_iso)
    video_seconds = int(video_duration.total_seconds())
    return video_seconds


def get_guardian_article(from_, to):
    MY_API_KEY = ""
    API_ENDPOINT = "https://content.guardianapis.com/search"
    my_params = {
        'from-date': from_,
        'to-date': to,
        'order-by': "newest",
        'show-fields': 'all',
        'section': 'artanddesign',
        'page-size': 200,
        'api-key': MY_API_KEY  
    }
    response = requests.get(API_ENDPOINT, params=my_params)

    if response.status_code == 200:
        data = response.json()
        pass
    else:
        print(f"Error: {response.status_code} - {response.text}")
    weekly_articles = data['response']['results'][:5]
    return weekly_articles

def get_video_items(article, number):
        api_service_name = "youtube"
        api_version = "v3"
        api_key = ""

        DEVELOPER_KEY = api_key
        youtube = googleapiclient.discovery.build(
                api_service_name, api_version, developerKey = DEVELOPER_KEY)
        request = youtube.search().list(
                part = "snippet",
                type = 'video',
                q =  get_keyword(article, number),
                order='date',
                maxResults=5
        )
        response = request.execute()
        video_ids = [item['id']['videoId'] for item in response['items']]
        video_request = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=",".join(video_ids)
        )
        video_response = video_request.execute()
        
        return video_response['items']


USE_LITE = True

if USE_LITE:
    MODEL_ID = "amazon.titan-text-lite-v1"  # We use Titan Lite for cost-effective text generation
    COST_PER_INPUT_TOKEN = 0.0003 / 1000  # $0.0003 per 1,000 input tokens
    COST_PER_OUTPUT_TOKEN = 0.0004 / 1000  # $0.0004 per 1,000 output tokens
    print(f"🚀 Using Bedrock model {MODEL_ID}! This is a fast and cheap, but not super accurate model.")
else:
    MODEL_ID = "amazon.titan-text-express-v1"  # We use Titan Express for more advanced text generation
    COST_PER_INPUT_TOKEN = 0.001 / 1000  # $0.001 per 1,000 input tokens
    COST_PER_OUTPUT_TOKEN = 0.0017 / 1000  # $0.0017 per 1,000 output tokens
    print(f"🚀 Using Bedrock model {MODEL_ID}! This is a not so cheap, but quite accurate model.")

pp = pprint.PrettyPrinter(indent=2)
# AWS Configuration
ROLE_ARN = "arn:aws:iam::870137400553:role/BedrockUserRole"  # The role we'll assume to access Bedrock
REGION = "us-east-1"  # Bedrock is currently only available in specific regions

def assume_role(role_arn: str, session_name: str) -> Dict[str, str]:
    print(f"🔐 Attempting to assume role: {role_arn}")

    try:
        if os.environ.get("AWS_PROFILE"):
            session = boto3.Session(profile_name=os.environ["AWS_PROFILE"])
        else:
            session = boto3.Session()

        sts_client = session.client("sts")
        assumed_role = sts_client.assume_role(RoleArn=role_arn, RoleSessionName=session_name)
        return assumed_role["Credentials"]

    except Exception as e:
        print(f"❌ Error assuming role: {str(e)}")
        raise

print("✅ Function assume_role defined!")


def calculate_token_count(text: str) -> int:
    return len(text) // 4


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    input_cost = input_tokens * COST_PER_INPUT_TOKEN
    output_cost = output_tokens * COST_PER_OUTPUT_TOKEN
    return input_cost + output_cost


def send_message_to_bedrock(message: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:

    try:
        credentials = assume_role(ROLE_ARN, "BedrockSession")
        session = boto3.Session(
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
        )
        bedrock_runtime = session.client(service_name="bedrock-runtime", region_name=REGION)
        payload = {
            "inputText": message,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "stopSequences": [],
                "temperature": temperature,
                "topP": 1,
            },
        }

        input_tokens = calculate_token_count(message)
        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        response_body = json.loads(response["body"].read())
        output_text = response_body["results"][0]["outputText"]
        output_tokens = calculate_token_count(output_text)
        total_cost = calculate_cost(input_tokens, output_tokens)

        print(f"💰 Total cost: ${total_cost:.6f}")

        return output_text

    except Exception as e:
        print(f"❌ Error sending message to Bedrock: {str(e)}")
        raise
weekly_articles = get_guardian_article("2024-12-09", "2024-12-13T12:00:00+05:00")
scripts = {}

for article in weekly_articles:
    web_title = article['webTitle']
    print(web_title)
    scripts[web_title] = {
        'headline': article['fields'].get('headline', ''),
        'bodytext': article['fields'].get('bodyText', ''),
        'date': article['fields'].get('firstPublicationDate', ''),
        'author': article['fields'].get('byline', ''),
        'youtube_details': {},
        'youtube_transcripts': {}
    }
    yt_video_items = get_video_items(article, 3)
    if not yt_video_items:
        yt_video_items = get_video_items(article, 2)
    
    videos_to_select = [{'id': video['id'], 'sentence':video['snippet']['description'][:100]} for video in yt_video_items]
    input = f"Choose one of the sentences from the list before that matches the artitle title. Sentences: {videos_to_select}. Article title: {web_title}. Return the sentence id."
    try:
        response = send_message_to_bedrock(input)
        response = response.strip()
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
    
    selected_video = next((video for video in videos_to_select if video['id'] in response), None)
    comprehend_client = boto3.client(service_name="comprehend", region_name="eu-west-1")
    comprehend_detect = ComprehendDetect(comprehend_client)
    
    if selected_video:
        try:
            key_phases = comprehend_detect.detect_languages(selected_video['sentence'])
            if key_phases[0]['LanguageCode'] != 'en': reponse = None
        except: pass
        print('🔢 Video id from Bedrock:', selected_video['id'])
    else:
        selected_video = {'id': None}
        
    sorted_list = sorted(yt_video_items, key=lambda x: x['statistics']['viewCount'])
    for video in sorted_list:
        try:
            lang = comprehend_detect.detect_languages(video['snippet']['description'])[0]['LanguageCode']
        except:
            lang = ""
        if lang != 'en':
            continue
        
        elif video['id'] == selected_video['id']:
            text = (
                video['snippet'].get('title', '') +
                video['snippet'].get('description', '') +
                str(video['snippet'].get('localized', '')) +
                ' '.join(video['snippet'].get('tags', []))
                )
            scripts[web_title]['youtube_details'] = text
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video['id'])
                scripts[web_title]['youtube_transcripts'] = re.sub(
                    r'\[.*?\]', '', ' '.join([item['text'] for item in transcript])
                )
            except Exception:
                continue
            continue
        elif int(video['statistics']['viewCount']) > 300 :
            text = (
                video['snippet'].get('title', '') +
                video['snippet'].get('description', '') +
                str(video['snippet'].get('localized', '')) +
                ' '.join(video['snippet'].get('tags', []))
                )
            scripts[web_title]['youtube_details'] = text
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video['id'])
                scripts[web_title]['youtube_transcripts'] = re.sub(
                    r'\[.*?\]', '', ' '.join([item['text'] for item in transcript])
                )
            except Exception:
                continue

def body_summary(original_body):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    input_text = original_body
    inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=100, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

for script in tqdm.tqdm(scripts.values(), desc="Processing bodytext", unit="item"):
    script['bodytext'] = body_summary(script['bodytext'])
    script['youtube_transcripts'] = body_summary(script['youtube_transcripts'])


podcast_prompt = f"""
You are a skilled podcast scriptwriter for the show “Art and Society: Weekly Digest,” which explores the intersection of art, culture, and social issues. Based on the following material (articles from The Guardian’s Art and Design section and related YouTube video transcripts), create a 5-minute podcast script that is around 700 words.

Instructions:
1. **Introduction (30 seconds)** – Start with a warm welcome, introduce the podcast, and briefly summarize the themes of the episode.
2. **Main Segment (4 minutes)** – Discuss 2–3 stories or ideas based on the provided data, weaving together insights from the articles and video transcripts. Include connections between art and societal issues, and highlight thought-provoking perspectives.
3. **Closing (30 seconds)** – Summarize the episode, encourage listener engagement (e.g., follow, share), and sign off.

Use a conversational but authoritative tone suitable for a podcast audience interested in art’s societal implications. Ensure the script is engaging, informative, and focused.

Here is the very raw data to guide your script:
{str(scripts)}

Summarize the data and make it a conversational podcast script. 

Please make sure to follow this structure and generate high-quality, detailed content. Avoid random, unrelated content and focus on providing insightful commentary.

The result should be around 700 words.
"""

try:
    podcast_response = send_message_to_bedrock(podcast_prompt) 
    print(podcast_response)   
except Exception as e:
    print(f"Test failed: {str(e)}")

polly_client = boto3.client('polly', region_name='eu-west-1')
response = polly_client.synthesize_speech(
    Text=podcast_response,
    OutputFormat='mp3',
    VoiceId='Joanna' 
)

with open('podcast.mp3', 'wb') as file:
    file.write(response['AudioStream'].read())