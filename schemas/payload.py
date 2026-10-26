from pydantic import BaseModel, Field

class Inputs(BaseModel):

    Review : str = Field(
        ...,
        max_length=1000,
        min_length=1,
    )
    Summary : str = Field(
        ...,
        max_length=50,
        min_length=1,
    )