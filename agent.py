import os

from transformers import pipeline
from smolagents import CodeAgent
from smolagents.agents import ChatMessage

from knowledge_base_tool import knowledge_base_search
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# HUGGING FACE PIPELINE WRAPPER
# ============================================================

class HFPipelineWrapper:

    def __init__(self, hf_pipeline):

        self.pipeline = hf_pipeline
        self.tokenizer = hf_pipeline.tokenizer


    def generate(self, messages, **kwargs):

        formatted_messages = []

        # --------------------------------------------------------
        # First convert Smolagents messages
        # --------------------------------------------------------

        for msg in messages:

            if isinstance(msg, ChatMessage):

                content = msg.content

                if isinstance(content, list):

                    text_parts = []

                    for item in content:

                        if isinstance(item, dict):

                            if "text" in item:
                                text_parts.append(
                                    str(item["text"])
                                )

                            elif "content" in item:
                                text_parts.append(
                                    str(item["content"])
                                )

                        else:
                            text_parts.append(
                                str(item)
                            )

                    content = "\n".join(text_parts)

                else:
                    content = str(content)

                formatted_messages.append({
                    "role": msg.role,
                    "content": content
                })


            elif isinstance(msg, dict):

                content = msg.get(
                    "content",
                    ""
                )

                if isinstance(content, list):

                    text_parts = []

                    for item in content:

                        if isinstance(item, dict):

                            if "text" in item:
                                text_parts.append(
                                    str(item["text"])
                                )

                            elif "content" in item:
                                text_parts.append(
                                    str(item["content"])
                                )

                        else:
                            text_parts.append(
                                str(item)
                            )

                    content = "\n".join(text_parts)

                else:
                    content = str(content)

                formatted_messages.append({
                    "role": msg.get(
                        "role",
                        "user"
                    ),
                    "content": content
                })


        # --------------------------------------------------------
        # Detect whether knowledge has already been retrieved
        # --------------------------------------------------------

        all_messages_text = "\n".join(
            str(message.get("content", ""))
            for message in formatted_messages
        )

        knowledge_retrieved = (
            "SOURCE 1" in all_messages_text
            or "SOURCE 2" in all_messages_text
            or "Document ID:" in all_messages_text
            or "Hybrid Score:" in all_messages_text
        )


        # --------------------------------------------------------
        # STEP 1 PROMPT
        # --------------------------------------------------------

        if not knowledge_retrieved:

            system_message = """

You are a knowledge-base agent.

You have exactly one tool:

knowledge_base_search(query)

The user has asked a knowledge-base question.

Your ONLY job in this step is to call the tool.

You MUST output valid Python code inside:

<code>
...
</code>

Use exactly this format:

<code>
knowledge_base_search(query="What is RAG?")
</code>

Do NOT answer the question yet.

Do NOT explain anything.

Do NOT provide a final answer.

First retrieve the information from the knowledge base.

"""


        # --------------------------------------------------------
        # STEP 2 PROMPT
        # --------------------------------------------------------

        else:

          
            system_message = """
You are a strictly grounded knowledge-base answer generator.

The knowledge base has already been searched.
Use the retrieved knowledge-base content in the conversation
to answer the user's question.

GROUNDING RULES:

1. Use only facts explicitly stated in the retrieved content.
2. Do not use pretrained knowledge, assumptions, or outside facts.
3. Do not add typical, common, or expected information unless
   it is explicitly present in the retrieved content.
4. Do not turn a possible benefit, implication, or example into
   a fact unless the source explicitly supports it.
5. Preserve the meaning and terminology of the source.
6. If the retrieved content does not answer the question,
   say that the knowledge base does not contain enough
   information to answer it.
7. Keep the answer concise.
8. Do not search again.

Before answering, check each factual statement:
- Can I point to a specific part of the retrieved content
  that supports this statement?
- If not, remove the statement.

OUTPUT FORMAT:
Return only a valid Python final_answer call inside a code block.

Example:
<code>
final_answer("The answer supported by the retrieved content.")
</code>

Do not output the answer as plain text outside the code block.
"""


        # --------------------------------------------------------
        # Add the appropriate system prompt
        # --------------------------------------------------------

        formatted_messages.insert(
            0,
            {
                "role": "system",
                "content": system_message
            }
        )


        # --------------------------------------------------------
        # Apply Qwen chat template
        # --------------------------------------------------------

        prompt = self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )


        # --------------------------------------------------------
        # Generation settings
        # --------------------------------------------------------

        generation_params = {

            "max_new_tokens": 128,

            "do_sample": False,

            "return_full_text": False
        }


        # --------------------------------------------------------
        # Generate response
        # --------------------------------------------------------


        print("\n===== DEBUG: MESSAGES SENT TO QWEN =====")

        for message in formatted_messages:
            print(f"\nROLE: {message['role']}")
        print(message["content"])

        print("\n===== END DEBUG MESSAGES =====\n")

        result = self.pipeline(
          prompt,
            max_length=None,
        **generation_params
)


        generated_text = result[0]["generated_text"]


        # --------------------------------------------------------
        # Handle different output formats
        # --------------------------------------------------------

        if isinstance(generated_text, list):

            text_parts = []

            for item in generated_text:

                if isinstance(item, dict):

                    text_parts.append(
                        str(
                            item.get(
                                "content",
                                item.get(
                                    "text",
                                    ""
                                )
                            )
                        )
                    )

                else:

                    text_parts.append(
                        str(item)
                    )

            generated_text = "\n".join(
                text_parts
            )


        generated_text = str(generated_text).strip()

# Remove Qwen end token if present
        generated_text = generated_text.replace(
    "<|im_end|>",
    ""
).strip()

# --------------------------------------------------------
# Ensure CodeAgent receives a valid <code> block
# --------------------------------------------------------

        if "<code>" not in generated_text:
            generated_text = f"<code>\nfinal_answer({generated_text!r})\n</code>"

        return ChatMessage(
    role="assistant",
    content=generated_text
)


# ============================================================
# CREATE HUGGING FACE PIPELINE
# ============================================================

print("Loading Qwen model...")


hf_pipeline = pipeline(

    "text-generation",

    model="Qwen/Qwen2.5-1.5B-Instruct",

    token=os.environ["HF_TOKEN"],

    trust_remote_code=True,

    # AMD Radeon laptop → CPU
    device=-1
)
hf_pipeline.model.generation_config.max_length = None

print("max_length:", hf_pipeline.model.generation_config.max_length)
print("max_new_tokens:", hf_pipeline.model.generation_config.max_new_tokens)


print("Qwen model loaded successfully!")


# ============================================================
# WRAP HUGGING FACE MODEL
# ============================================================

model = HFPipelineWrapper(
    hf_pipeline
)


print("Model wrapper created successfully!")


# ============================================================
# CREATE KNOWLEDGE BASE AGENT
# ============================================================

agent = CodeAgent(

    tools=[
        knowledge_base_search
    ],

    model=model,

    # Step 1 → search knowledge base
    # Step 2 → generate grounded answer
    max_steps=2,

    instructions="""
You are a knowledge-base question answering agent.

You have exactly one tool:

knowledge_base_search(query)

IMPORTANT RULES:

1. For every knowledge-base question, ALWAYS call
   knowledge_base_search first.

2. You MUST call the tool using Python code.

Example:

<code>
knowledge_base_search(query="What is RAG?")
</code>

3. After receiving the tool result, generate the final answer
   using the final_answer() function.

Example:

<code>
final_answer("RAG stands for Retrieval-Augmented Generation. It combines information retrieval with language generation.")
</code>

4. Use ONLY the information returned by the knowledge base.

5. Do NOT use your pretrained knowledge.

6. Do NOT invent facts or add information that is not present
   in the retrieved knowledge base context.

7. You may paraphrase the retrieved information to make the
   answer natural and easy to understand, but preserve the
   original meaning.

8. The final answer must be grounded in the retrieved
   knowledge base context.

9. Do NOT write the final answer as plain text outside a
   <code>...</code> block.

10. If the retrieved content does not contain enough information
    to answer the question, use:

<code>
final_answer("I don't have enough information in the knowledge base to answer this question.")
</code>

11. Do not call knowledge_base_search again after the first
    successful search.

12. Always provide the final answer using final_answer().
"""
)


print("Agent created successfully!")


# ============================================================
# FUNCTION FOR FRONTEND
# ============================================================

def ask_question(question: str) -> str:
    """
    Send a question to the Agentic RAG system
    and return the final answer.
    """

    result = agent.run(question)

    return str(result)
