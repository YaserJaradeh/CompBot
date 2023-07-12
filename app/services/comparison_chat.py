import asyncio
from typing import AsyncIterable, Tuple

from langchain import OpenAI
from langchain.agents import AgentExecutor, AgentType, create_json_agent
from langchain.agents.agent_toolkits.json.toolkit import JsonSpec, JsonToolkit
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chat_models import ChatOpenAI
from orkg import ORKG

from app.core.config import settings
from app.models.chat import AgentKind
from app.services.agents.handlers import AsyncStreamThoughtsAndAnswerHandler
from app.services.agents.pandas import create_custom_pandas_dataframe_agent

CHAT_MODELS = [
    "gpt-4",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
    "gpt-4-32k",
]


class ComparisonChatService:
    orkg_client = None

    def __init__(self, host=None):
        if host is None:
            host = settings.ORKG_HOST
        self.orkg_client = ORKG(host, simcomp_host=host + "simcomp")

    def initialize_agent(
        self,
        comparison_id: str,
        model: str,
        agent_kind: AgentKind,
        async_agent: bool = True,
    ) -> Tuple[AgentExecutor, AsyncStreamThoughtsAndAnswerHandler]:
        return self.create_agent(
            comparison_id=comparison_id,
            model=model,
            async_agent=async_agent,
            agent_kind=agent_kind,
        )

    def create_agent(
        self,
        comparison_id: str,
        model: str,
        agent_kind: AgentKind,
        async_agent: bool = True,
    ) -> Tuple[AgentExecutor, AsyncStreamThoughtsAndAnswerHandler]:
        """
        Create an agent that can access and use a large language model (LLM).

        Args:
            comparison_id: The ID of the comparison to use.
            async_agent: Whether to create an asynchronous agent or not.
            model: The name of the LLM to use.
            agent_kind: The type of agent to create.

        Returns:
            An agent that can access and use the LLM.
        """
        stream_handler = AsyncStreamThoughtsAndAnswerHandler()

        if model in CHAT_MODELS:
            llm_model = ChatOpenAI
        else:
            llm_model = OpenAI

        # Create an OpenAI object.
        llm = llm_model(
            streaming=True,
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=model,
        )

        comparison = self.orkg_client.contributions.compare_dataframe(
            comparison_id=comparison_id
        ).T

        if agent_kind == AgentKind.DATAFRAME:
            # Create a Pandas DataFrame agent.
            return (
                create_custom_pandas_dataframe_agent(
                    llm,
                    comparison,
                    verbose=settings.VERBOSE,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    agent_executor_kwargs={"handle_parsing_errors": True},
                    include_df_in_prompt=True,
                    return_intermediate_steps=False,
                    callback_manager=AsyncCallbackManager([stream_handler])
                    if async_agent
                    else None,
                    # max_execution_time=1
                ),
                stream_handler,
            )

        elif agent_kind == AgentKind.JSON:
            # Create a JSON agent.
            json_spec = JsonSpec(
                dict_=comparison.T.to_dict(),
                max_value_length=4000 if not isinstance(llm, ChatOpenAI) else 13000,
            )
            json_toolkit = JsonToolkit(spec=json_spec)
            return (
                create_json_agent(
                    llm,
                    json_toolkit,
                    verbose=settings.VERBOSE,
                    agent_executor_kwargs={"handle_parsing_errors": True},
                    callback_manager=AsyncCallbackManager([stream_handler])
                    if async_agent
                    else None,
                    max_execution_time=1,
                ),
                stream_handler,
            )
        else:
            raise NotImplementedError(f"Unknown agent kind: {agent_kind}")

    async def query_agent_async(
        self,
        comparison_id: str,
        query: str,
        model: str = None,
        agent_kind: AgentKind = AgentKind.DATAFRAME,
    ) -> AsyncIterable[str]:
        """
        Query an agent asynchronously and return the response as a string.

        Args:
            query: The query to ask the agent.
            comparison_id: The ID of the comparison to use.
            model: The model to use.
            agent_kind: The kind of agent to use.

        Returns:
            JSON responses from the agent, including the thoughts of the agent.
        """
        agent, prompt, stream_handler = self.initialize_components(
            comparison_id, query, model, agent_kind, True
        )

        task = asyncio.create_task(agent.arun(prompt))

        async for log in stream_handler.action_logs():
            yield log

        await task

    def query_agent(
        self,
        comparison_id: str,
        query: str,
        model: str = None,
        agent_kind: AgentKind = AgentKind.DATAFRAME,
    ) -> str:
        """
        Query an agent and return the response as a string.

        Args:
            query: The query to ask the agent.
            comparison_id: The ID of the comparison to use.
            model: The model to use.
            agent_kind: The kind of agent to use.

        Returns:
            The agents final answer as a json string.
        """
        agent, prompt, _ = self.initialize_components(
            comparison_id, query, model, agent_kind, False
        )

        # Run the prompt through the agent.
        response = agent.run(prompt)

        # Convert the response to a string.
        return response.__str__()

    def initialize_components(
        self, comparison_id, query, model, agent_kind, async_agent=True
    ) -> Tuple[AgentExecutor, str, AsyncStreamThoughtsAndAnswerHandler]:
        agent, handler = self.initialize_agent(
            comparison_id=comparison_id,
            async_agent=async_agent,
            model=model,
            agent_kind=agent_kind,
        )
        prompt = (
            """
            This is a table of data extracted from the ORKG that represents a comparison of several research papers.
            The rows are properties of the papers, and the columns are the papers (contributions) themselves.

            The questions will need you to look into the values, sometimes across multiple columns.
            The cells could contain multiple values and not just a single value.

            If there is a date in there you might need to parse it to find answers about the year or the month.

            If you do not know the answer, reply as follows:
            "Sorry!, I do not know."

            Return all output as a string.

            Lets think step by step.

            Below is the query.
            Query:
            """
            + query
        )
        return agent, prompt, handler
