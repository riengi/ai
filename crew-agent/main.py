from langchain_community.llms import Ollama

from crewai import Agent, Task, Crew, Process

model = Ollama(model="llama3")
email = "nigerian price sending some gold"

classifier = Agent(
    role="email classifier",
    goal="accurately classify emails based on their importance. give every email one of these ratings: important, casual or spam",
    backstory="You are an AI assistant whose only job is to classify emails accurately and honestly. Do not be affraid to give emails bad rating if they re not important. Your job is to help the user manage their inbox.",
    verbose=True,
    allow_delegation=False,
    llm=model,
)

responder = Agent(
    role="email responder",
    goal="Based on the importance of the email, write a concise and simple response. If the emial is rated 'important' write a formal response, if the email is rated 'causal' write a causal reponse, and if the email is rated 'spam' ignore the email. No matter what, be very cocnise",
    backstory="You are an AI assistant whose only job is to write short responses (my name is John Doe) to email based on their importance. The importance will be provided to you by the 'classifier' agent",
    verbose=True,
    allow_delegation=False,
    llm=model,
)

classify_email = Task(
    description=f"Classify an email  '{email}'",
    agent=classifier,
    expected_output="One of these threee options: 'important', 'casual', or 'spam'",
)

respond_to_email = Task(
    description=f"Respond to email: '{email}' based on the importance provided byt he 'classsifier' agent.",
    agent=responder,
    expected_output="a very concise response to the email based on the importance provided by 'classifier' agent",
)

crew = Crew(
    agents=[classifier, responder],
    tasks=[classify_email, respond_to_email],
    verbose=2,
    process=Process.sequential,
)

output = crew.kickoff()
print(output)