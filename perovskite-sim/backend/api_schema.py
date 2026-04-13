from pydantic import BaseModel, Field
from typing import List, Literal, Dict

class DeviceIn(BaseModel):
    V_bi: float
    Phi: float

class LayerIn(BaseModel):
    name: str
    role: str
    thickness: float
    eps_r: float
    mu_n: float
    mu_p: float
    ni: float
    N_A: float
    N_D: float
    D_ion: float
    P_lim: float
    P0: float
    tau_n: float
    tau_p: float
    n1: float
    p1: float
    B_rad: float
    C_n: float
    C_p: float
    alpha: float

class SimulateRequest(BaseModel):
    device: DeviceIn
    layers: List[LayerIn]
    sim_type: Literal['jv', 'impedance', 'degradation']
    sim_params: Dict[str, float] = Field(default_factory=dict)

class JVMetricsOut(BaseModel):
    V_oc: float
    J_sc: float
    FF: float
    PCE: float

class JVResultOut(BaseModel):
    V_fwd: List[float]
    J_fwd: List[float]
    V_rev: List[float]
    J_rev: List[float]
    metrics_fwd: JVMetricsOut
    metrics_rev: JVMetricsOut
    hysteresis_index: float

class ImpedanceResultOut(BaseModel):
    frequencies: List[float]
    Z_real: List[float]
    Z_imag: List[float]

class DegradationResultOut(BaseModel):
    times: List[float]
    PCE: List[float]
    V_oc: List[float]
    J_sc: List[float]

class JobStatusOut(BaseModel):
    status: Literal['pending', 'running', 'done', 'failed']
    result: JVResultOut | ImpedanceResultOut | DegradationResultOut | None = None
    error: str | None = None
