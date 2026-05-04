import os
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

api_key = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

# Upload file with purpose="ocr" and get a signed URL
file_path = "mistral7b.pdf"
print(f"Uploading {file_path}...")
with open(file_path, "rb") as f:
    uploaded = client.files.upload(
        file={"file_name": file_path, "content": f},
        purpose="ocr",
    )

signed_url = client.files.get_signed_url(file_id=uploaded.id)
print(f"File uploaded (id: {uploaded.id})")
print(f"Ask questions about the PDF (type 'quit' to exit).\n")

document_content = {
    "type": "document_url",
    "document_url": signed_url.url,
}

messages = []

while True:
    question = input("You: ").strip()
    if not question or question.lower() == "quit":
        break

    user_message = {"role": "user", "content": [{"type": "text", "text": question}]}

    # Only attach the document on the first message
    if not messages:
        user_message["content"].append(document_content)

    messages.append(user_message)

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=messages,
    )

    answer = response.choices[0].message.content
    print(f"\nAssistant: {answer}\n")

    messages.append({"role": "assistant", "content": answer})
