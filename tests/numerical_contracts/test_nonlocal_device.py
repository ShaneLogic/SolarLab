"""N02 preparation: manufactured algebra/metadata only, no physical call.

These tests do not evaluate source SG, Poisson, WKB, recycling or a device.
The separately admitted batch owns the physical value/direction/solve checks.
"""
import ast
import inspect
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from scripts.benchmarks.contract_prototype import ContractError, SECOND, VOLUME
from scripts.benchmarks.nonlocal_sparse import recycling_plan, wkb_plan
from scripts.benchmarks import nonlocal_device as witness


def small_plan(area=2.0):
    return witness.physical_graph([0, 1, 3, 6], area, recycling_plan(4, [range(4)]))


def manufactured_local(plan):
    """Invented linear coefficients solely for assembly/elimination tests."""
    n = plan.nodes
    matrix = 3 * witness.physical_storage(plan)
    for block, equation in enumerate(witness.CARRIER_ROWS):
        for row, node in zip(plan.rows(equation), plan.interior):
            matrix[row, block*n+node-1:block*n+node+2] += [-1, 2, -1]
            matrix[row, 2*n+node-1:2*n+node+2] += [-.2, .4, -.2]
    for row, node in zip(plan.rows("x_poisson"), plan.interior):
        matrix[row, 2*n+node-1:2*n+node+2] = [-1, 2, -1]
        matrix[row, node], matrix[row, n+node] = .3, -.3
    for variable, equation in (("n_m3", "z_n_contact"), ("p_m3", "zz_p_contact"),
                               ("phi_V", "y_phi_contact")):
        matrix[plan.rows(equation), plan.layout.offsets[variable].start+np.array([0,n-1])] = 1
    return matrix


def decimal_solve(matrix, rhs):
    with localcontext() as context:
        context.prec = 80
        rows = [[Decimal.from_float(float(x)) for x in row] + [Decimal.from_float(float(b))]
                for row, b in zip(matrix, rhs)]
        for j in range(len(rows)):
            pivot = max(range(j, len(rows)), key=lambda i: abs(rows[i][j]))
            rows[j], rows[pivot] = rows[pivot], rows[j]
            divisor = rows[j][j]
            assert divisor != 0
            rows[j] = [x/divisor for x in rows[j]]
            for i in range(len(rows)):
                if i != j:
                    multiplier = rows[i][j]
                    rows[i] = [a-multiplier*b for a,b in zip(rows[i],rows[j])]
        return np.array([float(row[-1]) for row in rows])


def test_physical_units_storage_and_nonuniform_half_cells():
    plan = small_plan()
    np.testing.assert_array_equal(plan.geometry.volumes, [1, 3, 5, 3])
    assert sum(plan.geometry.volumes) == plan.area_m2*(plan.x[-1]-plan.x[0])
    mass = witness.physical_storage(plan)
    assert mass.shape == (12,12) and np.count_nonzero(mass) == 4
    for block, row in enumerate(witness.CARRIER_ROWS):
        np.testing.assert_array_equal(mass[plan.rows(row), block*4+plan.interior], [3,5])
    for row in ("x_poisson", "y_phi_contact", "z_n_contact", "zz_p_contact"):
        assert not np.any(mass[plan.rows(row)])
    term = next(t for t in plan.terms if t.id == "local:a_n:n_m3")
    assert term.unit == VOLUME/SECOND


def test_FV_signs_geometry_once_and_boundary_reservoir_ledger():
    plan = small_plan()
    y = np.ones(12)
    rate = np.zeros(12)
    rate[[1,2,5,6]] = [2,3,4,5]
    result = witness.finite_volume_rows(plan,y,rate,charge_C=2,
        local_currents_A_m2=[[2,4,8],[3,9,12]],
        net_local_generation_m3_s=[1,2,3,4], displacement_C_m2=[5,8,10],
        charge_density_C_m3=[1,2,3,4], contacts={key:[1,1] for key in witness.VARIABLES})
    np.testing.assert_array_equal(result['residual'][plan.rows('a_n')],[-2,-4])
    np.testing.assert_array_equal(result['residual'][plan.rows('b_p')],[12,13])
    np.testing.assert_array_equal(result['residual'][plan.rows('x_poisson')],[0,-11])
    np.testing.assert_array_equal(result['exterior_displacement_C_m2'],[4.5,16])
    np.testing.assert_array_equal(result['reservoir_exchange_into_device_particle_s'],[[-3,-4],[2,-24]])
    for block,equation in enumerate(witness.CARRIER_ROWS):
        balance = plan.geometry.volumes @ rate[block*4:(block+1)*4] - 34
        balance -= sum(result['reservoir_exchange_into_device_particle_s'][block])
        assert balance == sum(result['residual'][plan.rows(equation)])


def test_component_rate_and_face_divergence_have_different_unit_lifts():
    plan=small_plan(); scales=np.arange(1,13,dtype=float)
    value=np.ones(12); jacobian=np.ones((12,12))
    rate,rate_j=witness.component_rows(plan,value,jacobian,scales,7,'recycling_rate_density')
    face,face_j=witness.component_rows(plan,value,jacobian,scales,7,'wkb_face_divergence')
    rows=plan.rows('a_n')
    np.testing.assert_array_equal(rate[rows],[-21,-35])
    np.testing.assert_array_equal(face[rows],[-14,-14])
    np.testing.assert_allclose(rate_j[rows]*scales,-np.array([21,35])[:,None]*np.ones((2,12)))
    np.testing.assert_allclose(face_j[rows]*scales,-14*np.ones((2,12)))
    assert not np.any(face[plan.rows('x_poisson')])


def test_zero_coefficients_keep_nonlocal_slots_and_subnormal_loss_is_rejected():
    plan=small_plan(); local=manufactured_local(plan); coupling=np.zeros((12,12))
    assembled=witness.assemble(plan,local,{'recycling.0':coupling},topology=plan.topology)
    assert assembled.nnz == plan.graph.nnz > assembled.count_nonzero()
    row=plan.rows('a_n')[0]; coupling[row,7]=.125
    changed=witness.assemble(plan,local,{'recycling.0':coupling},topology=plan.topology)
    assert changed[row,7] == .125 and changed.nnz == assembled.nnz
    coupling[row,11]=np.nextafter(0.,1.)
    with pytest.raises(ContractError,match='outside_declared'):
        witness.assemble(plan,local,{'recycling.0':coupling},topology=plan.topology)
    with pytest.raises(ContractError,match='complete_nonlocal'):
        witness.assemble(plan,local,{},topology=plan.topology)


def test_scaled_recycling_dense_sparse_auxiliary_and_rowwise_error():
    plan=small_plan(); n=plan.nodes
    local=manufactured_local(plan)
    raw_u=np.zeros((3*n,1)); raw_u[:2*n]=.001
    raw_v=np.zeros((3*n,1)); raw_v[:2*n,0]=np.arange(1,2*n+1)/10
    state_scales=np.arange(1,3*n+1,dtype=float)
    u,v=witness.recycling_border(plan,raw_u,raw_v,state_scales,7)
    assembled=witness.assemble(plan,local,{'recycling.0':u@v.T},topology=plan.topology)
    rows=np.arange(1,3*n+1,dtype=float); cols=1/rows; residual=rows/7
    matrix,rhs=witness.normalized_system(assembled,residual,rows,cols)
    expected=decimal_solve(matrix,rhs)
    sparse=splu(csc_matrix(matrix)).solve(rhs)
    auxiliary=witness.auxiliary_system(plan,local,u,v,rows,cols,np.array([3.]))
    inactive=witness.auxiliary_system(plan,local,np.zeros_like(u),np.zeros_like(v),rows,cols,np.array([3.]))
    assert inactive.nnz == auxiliary.nnz and inactive.nnz > inactive.count_nonzero()
    bordered=splu(auxiliary).solve(np.r_[rhs,0.])
    np.testing.assert_allclose(sparse,expected,rtol=2e-13,atol=2e-14)
    np.testing.assert_allclose(bordered[:3*n],expected,rtol=2e-13,atol=2e-14)
    diagnostics=witness.solve_diagnostics(matrix,rhs,sparse)
    assert diagnostics['maximum_componentwise_backward_error'] < 1e-14
    # A large row can hide a small-row failure from a global normwise metric.
    diagnostics=witness.solve_diagnostics(np.diag([1e20,1.]),[1e20,1.],[1.,2.])
    assert diagnostics['normwise_backward_error'] < 1e-19
    assert diagnostics['maximum_componentwise_backward_error'] > .3


def test_WKB_full_phi_and_eliminated_Poisson_remote_hole_support():
    component=wkb_plan(13,(0,4,8,13),[(3,8),(3,8),(3,7),(3,7)],(3,12,4,0))
    plan=witness.physical_graph(np.arange(13.),1.,component)
    assert set(range(26,39)) <= set(component.couplings[0].columns)
    assert not set(range(13,26)) & set(component.couplings[0].columns)
    support=witness.poisson_chain_support(plan)
    carrier_row=plan.rows('a_n')[7]
    reduced_row=list(support['retained_rows']).index(carrier_row)
    assert (reduced_row,14) in set(map(tuple,support['chain_edges']))
    assert not support['coefficient_mask_used']
    full=manufactured_local(plan); rhs=np.arange(1,40,dtype=float)
    calls=[]
    def block_solve(a,b):
        calls.append((a.shape,b.shape)); return np.linalg.solve(a,b)
    reduced=witness.poisson_schur(plan,full,rhs,block_solve)
    carriers=np.linalg.solve(reduced['matrix'],reduced['rhs'])
    phi=reduced['phi_per_carrier']@carriers+reduced['phi_rhs']
    np.testing.assert_allclose(np.r_[carriers,phi],decimal_solve(full,rhs),rtol=2e-13,atol=2e-13)
    assert calls == [((13,13),(13,27))]
    with pytest.raises(ContractError,match='only_recycling'):
        witness.recycling_border(plan,np.zeros((39,1)),np.zeros((39,1)),np.ones(39),1)


def test_WKB_branch_change_invalidates_even_equal_union_edges():
    a=wkb_plan(13,(0,4,8,13),[(3,8),(3,8),(3,7),(3,7)],(3,12,4,0))
    b=wkb_plan(13,(0,4,8,13),[(3,8),(3,7),(3,7),(3,7)],(3,12,4,0))
    pa=witness.physical_graph(np.arange(13.),1.,a)
    pb=witness.physical_graph(np.arange(13.),1.,b)
    np.testing.assert_array_equal(pa.graph.edges(),pb.graph.edges())
    assert pa.topology != pb.topology
    with pytest.raises(ContractError,match='structural_recompile'):
        witness.assemble(pa,np.zeros((39,39)),{'wkb.electron':np.zeros((39,39))},topology=pb.topology)


@pytest.mark.parametrize('field',['source_request','artifact','state','geometry','topology'])
def test_cached_source_and_state_are_all_required(field):
    expected={key:'a'*64 for key in ('source_request','artifact','state','geometry','topology')}
    witness.verify_component_binding(expected,expected.copy())
    changed=expected.copy(); changed[field]='b'*64
    with pytest.raises(ContractError,match='exact_component'):
        witness.verify_component_binding(expected,changed)


def test_physical_entry_is_denied_before_any_physical_import_or_callback():
    plan=small_plan(); state=np.ones(12)
    with pytest.raises(ContractError,match='admitted_counted'):
        witness.source_local_fields(plan,{},state,None)
    with pytest.raises(ContractError,match='same_state_material'):
        witness.source_local_jacobian(plan,{},state,1,{},lambda name:pytest.fail(name))
    def deny(name):
        raise PermissionError(name)
    with pytest.raises(PermissionError,match='local_constitutive_point'):
        witness.source_local_fields(plan,{},state,deny)
    module=ast.parse(inspect.getsource(witness))
    top_imports=[node for node in module.body if isinstance(node,(ast.Import,ast.ImportFrom))]
    assert not any(isinstance(node,ast.ImportFrom) and (node.module or '').startswith('perovskite_sim') for node in top_imports)
    functions={node.name:node for node in module.body if isinstance(node,ast.FunctionDef)}
    body=ast.unparse(functions['source_local_jacobian'])
    assert 'sg_fluxes_n_jacobian' in body and 'sg_fluxes_p_jacobian' in body
    assert 'SG_jacobian_with_flux' in body and 'coupled_device_prototype' not in body
    native=ast.parse((Path(__file__).parents[2]/'perovskite-sim/perovskite_sim/discretization/fe_operators.py').read_text())
    declarations={node.name:node for node in native.body if isinstance(node,ast.FunctionDef)}
    for name,density,diffusion in [('sg_fluxes_n_jacobian','n','D_n'),('sg_fluxes_p_jacobian','p','D_p')]:
        assert [arg.arg for arg in declarations[name].args.args] == ['phi',density,'dx',diffusion,'V_T']


def test_2D_support_is_cartesian_absorber_not_a_flattened_trapezoid():
    plan=witness.recycling_2d_declaration(3,4,[(1,3)])
    assert plan.nodes == 12
    assert plan.couplings[0].rows == tuple(range(3,9))+tuple(range(15,21))
    assert plan.couplings[0].columns == plan.couplings[0].rows


def test_complete_nonlocal_endpoint_ledger_closes_manufactured_particle_balance():
    plan=small_plan();state=np.ones(12)
    result=witness.finite_volume_rows(plan,state,np.zeros(12),charge_C=2,
        local_currents_A_m2=np.zeros((2,3)),net_local_generation_m3_s=np.zeros(4),
        displacement_C_m2=np.zeros(3),charge_density_C_m3=np.zeros(4),
        contacts={key:[1,1] for key in witness.VARIABLES},recycling_m3_s=[1,2,3,4],
        nonlocal_currents_A_m2=[[2,4,8],[0,0,0]])
    np.testing.assert_array_equal(result['residual'][plan.rows('a_n')],[-8,-19])
    np.testing.assert_array_equal(result['residual'][plan.rows('b_p')],[-6,-15])
    np.testing.assert_array_equal(result['reservoir_exchange_into_device_particle_s'],[[-3,-4],[-1,-12]])
    for block,equation in enumerate(witness.CARRIER_ROWS):
        assert sum(result['residual'][plan.rows(equation)]) == -34-sum(result['reservoir_exchange_into_device_particle_s'][block])
