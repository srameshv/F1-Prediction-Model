from pydantic import BaseModel

class RaceInput(BaseModel):
    raceId: int
    driverId: int
    constructorId: int
    grid: int
    points: float
    positionOrder: int
    wins_season: int
    team_points_avg: float
    driver_points_last3: float
    circuit_type: str
    weather: str
