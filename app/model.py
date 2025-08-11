from pydantic import BaseModel
from typing import Literal, Union
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

    def cast_to_openai_schema(
        self,
    ) -> Union[
        ChatCompletionUserMessageParam,
        ChatCompletionAssistantMessageParam,
        ChatCompletionSystemMessageParam,
    ]:
        """OpenAIのAPIに渡す形式に変換する"""
        if self.role == "user":
            return ChatCompletionUserMessageParam(role="user", content=self.content)
        elif self.role == "assistant":
            return ChatCompletionAssistantMessageParam(
                role="assistant", content=self.content
            )
        else:
            return ChatCompletionSystemMessageParam(role="system", content=self.content)
        