import logging
import json
from datetime import datetime

import arxiv
from openai import OpenAI
from slack_sdk import WebClient
from jinja2 import Template
from slack_sdk.errors import SlackApiError
from aws_lambda_powertools.utilities.typing import LambdaContext

from app.model import Message
from app.env import env

logger = logging.getLogger()
logger.setLevel("INFO")

CATEGORIES = {
    "cs.AI",
    "cs.CV",
}

SYSTEM_PROMPT = """
### 指示 ###
論文の内容を理解した上で，重要なポイントを箇条書きで3点書いてください。

### 箇条書きの制約 ###
- 最大3個
- 日本語

### 対象とする論文の内容 ###
{{article_text}}

### 出力形式 ###
タイトル(和名)

- 箇条書き1
- 箇条書き2
- 箇条書き3
"""

client = arxiv.Client()
slack_client = WebClient(token=env.SLACK_TOKEN)
llm_client = OpenAI(
    api_key=env.GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/",
)


def handler(event: dict, context: LambdaContext) -> dict[str, object]:
    search = arxiv.Search(
        query=env.ARXIV_QUERY,
        max_results=env.ARXIV_MAX_RESULTS * 3,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    results = client.results(search)

    result_list = []
    for result in results:
        if len((set(result.categories) & CATEGORIES)) == 0:
            continue

        if len(result_list) >= env.ARXIV_MAX_RESULTS:
            break

        result_list.append(result)

    if len(result_list) == 0:
        response = slack_client.chat_postMessage(
            channel=env.SLACK_CHANNEL,
            text=f"{'=' * 40}\n{env.ARXIV_QUERY}に関する論文は有りませんでした！\n{'=' * 40}",
        )
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "No results found",
                }
            ),
        }

    # 週の開始メッセージを送信
    today = datetime.now()
    week_start_message = f"""📚 今週の論文通知を開始します
{'=' * 50}
日付: {today.strftime('%Y年%m月%d日 (%A)')}
検索クエリ: {env.ARXIV_QUERY}
見つかった論文数: {len(result_list)}件
{'=' * 50}"""
    
    slack_client.chat_postMessage(
        channel=env.SLACK_CHANNEL,
        text=week_start_message
    )

    for i, result in enumerate(result_list, start=1):
        try:
            message = f"{env.ARXIV_QUERY}: {i}本目\n" + get_summary(result)

            response = slack_client.chat_postMessage(
                channel=env.SLACK_CHANNEL, text=message
            )
            logger.info(f"Message posted: {response['ts']}")

        except SlackApiError as e:
            logger.error(f"Error posting message: {e}")
            return {
                "statusCode": 500,
                "body": json.dumps(
                    {
                        "message": "Error posting message",
                    }
                ),
            }

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": "Success",
            }
        ),
    }


def get_summary(result: arxiv.Result) -> str:
    text = f"title: {result.title}\nbody: {result.summary}"
    system_prompt = Template(SYSTEM_PROMPT).render(article_text=text)
    prompt = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=text),
    ]
    response = llm_client.chat.completions.create(
        model="gemini-2.5-flash",
        n=1,
        messages=[message.cast_to_openai_schema() for message in prompt],
    )

    summary = response.choices[0].message.content
    title_en = result.title
    title, *body = summary.split("\n")
    body = "\n".join(body)
    date_str = result.published.strftime("%Y-%m-%d %H:%M:%S")
    message = f"発行日: {date_str}\n{result.entry_id}\n{title_en}\n{title}\n{body}\n"

    return message

if __name__ == "__main__":
    handler({}, {})