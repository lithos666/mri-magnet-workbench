import unittest
import tempfile
import numpy as np
from mri_magnet.active_set import active_set_polish
from mri_magnet.objectives import feasibility_score


class BalancedLPTests(unittest.TestCase):
    def test_budgeted_solver_relinearizes_after_acceptance_without_hidden_fields(self):
        from mri_magnet.optimization import run_optimization
        calls=[]
        def fields(design):
            x=design['x'];mean=.04+.004*x;spread=1e-5+2e-6*x
            calls.append(x)
            return np.array([[0,0,mean-spread/2],[0,0,mean+spread/2]])
        req={'min_mean_norm_T':.04,'scoring':{'mode':'balanced'}}
        variables=[{'name':'x','lower':0.,'upper':1.,'initial':.4}]
        with tempfile.TemporaryDirectory() as folder:
            result=run_optimization(variables,lambda v:{'x':float(v[0])},None,
                lambda d:{'valid':True},{'x':.4},folder,max_evaluations=24,
                population_size=5,maxiter=0,requirements=req,field_values=fields,
                field_weights=[.5,.5],active_set_evaluations=18,active_set_trust_radius=.03)
        info=result['metadata']['rounds'][0]['active_set_polish']
        self.assertGreaterEqual(info['linearization_passes'],2)
        self.assertGreaterEqual(info['accepted_steps'],2)
        self.assertEqual(result['metadata']['field_evaluations'],len(calls))
        self.assertLessEqual(len(calls),24)

    def test_lp_uses_field_reward_but_accepts_only_actual_balanced_improvement(self):
        req={'min_mean_norm_T':.04,'max_pp_norm_ppm':None,
             'scoring':{'mode':'balanced','field_weight':.5,'field_gain_scale':.1,'ppm_scale':1000.}}
        calls=[]
        def query(x):
            mean=.04+.004*x[0];spread=1e-5+2e-6*x[0]
            fields=np.array([[0,0,mean-spread/2],[0,0,mean+spread/2]])
            m={'mean_norm_T':mean,'pp_norm_ppm':spread/mean*1e6}
            calls.append(float(x[0]))
            return {'fields':fields,'metrics':m,'score':feasibility_score(m,req)}
        start=np.array([.4]);before=query(start)
        result,info=active_set_polish(start,[0.],[1.],[False],query,
            remaining=lambda:20-len(calls),weights=[.5,.5],requirements=req,trust_radius=.2)
        self.assertEqual(info['accepted_steps'],1)
        self.assertGreater(result[0],start[0])
        self.assertLess(query(result)['score'],before['score'])
        self.assertIn('balanced',info['linearized_scoring'])

if __name__=='__main__':unittest.main()
