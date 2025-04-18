from typing import Any, Dict, List

from pydantic import BaseModel, Field


class DocumentField(BaseModel):
    field_type: str
    label: str
    data: Dict[str, Any]


class Document(BaseModel):
    title: str
    content: List[DocumentField] = Field(default_factory=list)

    def get_text_content(self) -> str:
        text_parts = []
        for field in self.content:
            if field.field_type == "text_area":
                data = field.data.get("value", "")
                if isinstance(data, str):
                    text_parts.append(data)
        return " ".join(text_parts)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "field_count": len(self.content),
            "field_types": [field.field_type for field in self.content],
        }
