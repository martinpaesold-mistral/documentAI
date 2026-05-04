import base64
import os
from dotenv import load_dotenv
from mistralai import Mistral
from mistralai.models import ResponseFormat, JSONSchema

load_dotenv()

api_key = os.environ.get("MISTRAL_API_KEY")

client = Mistral(api_key=api_key)

def encode_file(file_path):
    with open(file_path, "rb") as pdf_file:
        return base64.b64encode(pdf_file.read()).decode('utf-8')

file_path = "mistral7b.pdf"
base64_file = encode_file(file_path)

ocr_response = client.ocr.process(
    document={
      "type": "document_url",
      "document_url": f"data:application/pdf;base64,{base64_file}"
    },
    model="mistral-ocr-latest",
	include_image_base64=True,
	bbox_annotation_format=ResponseFormat(
		type="json_schema",
		json_schema=JSONSchema(
			name="image_annotation_schema",
			schema_definition={
				"type": "object",
				"required": ["description", "category"],
				"properties": {
					"description": {"type": "string"},
					"category": {
						"type": "string",
						"enum": ["figure", "table", "diagram", "chart", "photo", "other"]
					},
					"caption": {"type": "string"},
				}
			},
			strict=True,
		),
	),
	document_annotation_format=ResponseFormat(
		type="json_schema",
		json_schema=JSONSchema(
			name="response_schema",
			schema_definition={
				"type": "object",
				"required": [
					"title",
					"authors",
					"abstract",
					"introduction",
					"architectural_details",
					"results",
					"instruction_finetuning",
					"guardrails",
					"conclusion",
					"references"
				],
				"properties": {
					"title": {
						"type": "string"
					},
					"authors": {
						"type": "array",
						"items": {
							"type": "string"
						}
					},
					"abstract": {
						"type": "object",
						"required": [
							"content"
						],
						"properties": {
							"content": {
								"type": "string"
							}
						}
					},
					"introduction": {
						"type": "object",
						"required": [
							"content"
						],
						"properties": {
							"content": {
								"type": "string"
							}
						}
					},
					"architectural_details": {
						"type": "object",
						"required": [
							"sliding_window_attention",
							"rolling_buffer_cache",
							"prefill_and_chunking",
							"model_architecture"
						],
						"properties": {
							"sliding_window_attention": {
								"type": "string"
							},
							"rolling_buffer_cache": {
								"type": "string"
							},
							"prefill_and_chunking": {
								"type": "string"
							},
							"model_architecture": {
								"type": "string"
							},
							"figure_1": {
								"type": "string"
							},
							"figure_2": {
								"type": "string"
							},
							"figure_3": {
								"type": "string"
							}
						}
					},
					"results": {
						"type": "object",
						"required": [
							"benchmark_categories",
							"detailed_results",
							"performance_comparison",
							"size_and_efficiency",
							"evaluation_differences"
						],
						"properties": {
							"benchmark_categories": {
								"type": "string"
							},
							"detailed_results": {
								"type": "string"
							},
							"performance_comparison": {
								"type": "string"
							},
							"size_and_efficiency": {
								"type": "string"
							},
							"evaluation_differences": {
								"type": "string"
							},
							"table_2": {
								"type": "string"
							},
							"figure_4": {
								"type": "string"
							},
							"figure_5": {
								"type": "string"
							}
						}
					},
					"instruction_finetuning": {
						"type": "object",
						"required": [
							"content",
							"table_3",
							"human_evaluation"
						],
						"properties": {
							"content": {
								"type": "string"
							},
							"table_3": {
								"type": "string"
							},
							"human_evaluation": {
								"type": "string"
							},
							"figure_6": {
								"type": "string"
							}
						}
					},
					"guardrails": {
						"type": "object",
						"required": [
							"system_prompt_enforcement",
							"content_moderation"
						],
						"properties": {
							"system_prompt_enforcement": {
								"type": "string"
							},
							"content_moderation": {
								"type": "string"
							},
							"table_4": {
								"type": "string"
							},
							"table_5": {
								"type": "string"
							}
						}
					},
					"conclusion": {
						"type": "object",
						"required": [
							"content"
						],
						"properties": {
							"content": {
								"type": "string"
							}
						}
					},
					"references": {
						"type": "array",
						"items": {
							"type": "string"
						}
					},
					"document_url": {
						"type": "string"
					}
				}
			},
			strict=True,
		),
	),
	table_format="markdown",
)

import json

output = {}
output["pages"] = []
for page in ocr_response.pages:
    page_data = {"index": page.index, "markdown": page.markdown, "images": []}
    if page.dimensions:
        page_data["dimensions"] = {
            "width": page.dimensions.width,
            "height": page.dimensions.height,
            "dpi": page.dimensions.dpi,
        }
    for img in page.images:
        img_data = {
            "id": img.id,
            "bounding_box": {
                "top_left_x": img.top_left_x,
                "top_left_y": img.top_left_y,
                "bottom_right_x": img.bottom_right_x,
                "bottom_right_y": img.bottom_right_y,
            },
        }
        if img.image_annotation:
            img_data["annotation"] = json.loads(img.image_annotation)
        page_data["images"].append(img_data)
    output["pages"].append(page_data)

if ocr_response.document_annotation:
    output["document_annotation"] = json.loads(ocr_response.document_annotation)

with open("output.json", "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Output saved to output.json")
print(f"Total images found: {sum(len(p['images']) for p in output['pages'])}")
