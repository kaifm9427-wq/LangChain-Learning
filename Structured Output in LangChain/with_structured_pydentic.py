from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated , Optional, Literal
from pydantic import BaseModel,Field

load_dotenv()

model=ChatGoogleGenerativeAI(
    model='gemini-2.5-flash'
)

#schema for data
class Review(BaseModel): # Review class typeddict se inherit karega

    key_themes: list[str]=Field(description="write down all key themes from review in a list ")
    summary: list[str]=Field(description="A brief summary of the review")
    sentiment: Literal["pos","neg"]=Field(description="You need to provide seniments either negative, positive or neutral")
    pros: list[str]=Field(description="write all pros inside a list")
    cons: Optional[list[str]]=Field(description="write all cons inside a list")
    name: Optional[str]= Field(description="write the name of the reviewer") 

structured_model = model.with_structured_output(Review)

result=structured_model.invoke("""

The smartphone offers excellent hardware performance with a powerful processor that handles gaming, multitasking, and heavy applications smoothly. The display is bright, color-accurate, and provides a fluid high refresh rate experience, making content consumption and daily usage enjoyable. Battery life is another strong point, easily lasting a full day with fast charging support that minimizes downtime.

The camera system performs well in daylight with detailed photos, balanced colors, and good dynamic range. Video stabilization is decent, and portrait shots come out sharp in most conditions. The build quality also feels premium, giving the device a solid and modern look that competes well in its price segment.

However, the software experience is one of the weaker aspects of the phone. There are too many pre-installed applications, some of which cannot be removed, making the UI feel cluttered and less optimized. The interface design also feels outdated compared to cleaner Android skins offered by competing brands. Occasional ads, duplicate apps, and inconsistent animations further reduce the overall user experience.

Another downside is the uncertainty around long-term software updates and optimization. While the hardware is future-ready, delayed updates and inconsistent software support may impact performance and security over time. Low-light camera performance is average, and heating can occasionally appear during extended gaming sessions.

Pros:

* Powerful performance for gaming and multitasking
* Excellent display quality with smooth refresh rate
* Strong battery backup with fast charging
* Premium build quality and attractive design
* Good daylight camera performance

Cons:

* Excessive bloatware and unnecessary apps
* Outdated and cluttered UI experience
* Average low-light photography
* Occasional heating during heavy usage
* Software updates could be faster and more reliable
review by Mohammed Kaif

""")

print(result.name)
