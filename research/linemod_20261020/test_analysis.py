"""CPU regression fixtures; no model inference or installed-package changes."""
import csv,json,tempfile,unittest
from pathlib import Path
import numpy as np
import trimesh,yaml,cv2
import analyze as a
import extended as e
import supervise as supervisor
import finalize as finalizer

class AnalysisTests(unittest.TestCase):
    def test_directional_adi(self):
        gt=np.array([[0.,0,0],[1,0,0],[3,0,0]])
        pred=np.array([[1.,0,0],[1,1,0],[1,3,0]])
        _,adi=a.vertex_errors(pred,gt)
        self.assertAlmostEqual(adi,1.)
        _,reverse=a.vertex_errors(gt,pred)
        self.assertAlmostEqual(reverse,4/3)

    def test_bop_distance_and_symmetry_composition(self):
        pts=np.array([[0.,0,0],[10,0,0],[0,15,3]])
        R=np.array([[1.,0,0],[0,0,-1],[0,1,0]]);t=np.array([0.,0,1000]);K=np.array([[100.,0,320],[0,100,240],[0,0,1]])
        self.assertEqual(e.distances(pts,R,t,R,t,K,[np.eye(4)]),(0.,0.))
        s=np.diag([-1.,-1,1,1]);s[:3,3]=[2.,3,0]
        self.assertTrue(np.allclose(e.distances(pts,R@s[:3,:3],R@s[:3,3]+t,R,t,K,[np.eye(4),s]),[0.,0.]))
        mssd,mspd=e.distances(pts,np.eye(3),t+[10,0,0],np.eye(3),t,K,[np.eye(4)])
        self.assertAlmostEqual(mssd,10.);self.assertAlmostEqual(mspd,1.)

    def test_full_pipeline(self):
        old=(a.OUT,a.DATA,a.ROOT,a.PRED,a.IDS)
        root=Path(tempfile.mkdtemp(prefix='foundationpose-analysis-test-'))
        try:
            a.OUT=root;a.DATA=root/'data';a.ROOT=root/'repo';a.PRED=root/'pred.yml';a.IDS=[1,10]
            (a.DATA/'models').mkdir(parents=True);(a.DATA/'models_eval').mkdir();(a.ROOT/'weights').mkdir(parents=True)
            for f in ['run_linemod.py','datareader.py','estimater.py']:(a.ROOT/f).write_text('test fixture')
            info={};pred={}
            for obj in a.IDS:
                scene=a.DATA/f'lm_test_all/test/{obj:06d}'
                for name in ['rgb','depth','mask_visib']:(scene/name).mkdir(parents=True)
                good=np.eye(4);good[2,3]=1;shift=good.copy();shift[0,3]=.1;bad=good.copy();bad[0,0]=2
                pred[obj]={'000000':{obj:good.tolist()},'000001':{obj:shift.tolist()},'000002':{obj:bad.tolist()},'000004':{obj:'malformed'}}
                gt={};camera={}
                for im in range(5):
                    cv2.imwrite(str(scene/f'rgb/{im:06d}.png'),np.zeros((480,640,3),np.uint8))
                    for name in ['depth','mask_visib']:
                        filename=f'{im:06d}_000000.png' if name=='mask_visib' else f'{im:06d}.png'
                        cv2.imwrite(str(scene/name/filename),np.ones((480,640),np.uint16))
                    gt[str(im)]=[dict(obj_id=obj,cam_R_m2c=np.eye(3).reshape(-1).tolist(),cam_t_m2c=[0,0,1000])]
                    camera[str(im)]={'cam_K':[500,0,320,0,500,240,0,0,1],'depth_scale':1}
                (scene/'scene_gt.json').write_text(json.dumps(gt));(scene/'scene_camera.json').write_text(json.dumps(camera))
                (scene/'scene_gt_info.json').write_text(json.dumps({str(im):[dict(visib_fract=1.,px_count_visib=2500)] for im in range(5)}))
                mesh=trimesh.creation.box(extents=[100,100,100])
                for name in ['models','models_eval']:mesh.export(a.DATA/f'{name}/obj_{obj:06d}.ply')
                info[str(obj)]={'diameter':173.205}
            for name in ['models','models_eval']:(a.DATA/name/'models_info.json').write_text(json.dumps(info))
            a.PRED.write_text(yaml.safe_dump(pred));a.analyze();e.run()
            with open(root/'summary.csv') as f:rows=list(csv.DictReader(f))
            pooled=rows[-1];self.assertEqual([pooled[k] for k in ['expected','valid','missing','invalid']],['10','4','2','4'])
            self.assertAlmostEqual(float(pooled['adds_recall_0.1d']),.2)
            with open(root/'per_pose.csv') as f:per=list(csv.DictReader(f))
            self.assertEqual(float(per[0]['translation_mm']),0.)
            self.assertAlmostEqual(float(per[1]['translation_mm']),100.)
            self.assertFalse(json.loads((root/'prediction_validation.json').read_text())['coverage_complete'])
            self.assertTrue((root/'examples/obj_01_success_000000.png').exists())
            self.assertTrue((root/'examples/obj_01_failure_000001.png').exists())
            with open(root/'dataset_inventory.csv') as f:inventory=list(csv.DictReader(f))
            self.assertTrue(all(r['missing_depth']=='0' and r['missing_masks']=='0' for r in inventory))
            previous_out=supervisor.OUT
            try:
                supervisor.OUT=root
                (root/'REPORT.md').write_text('Synthetic fixture report\n')
                supervisor.completed_report();supervisor.completed_report()
                report=(root/'REPORT.md').read_text()
                self.assertEqual(report.count('<!-- AUTO_MEASURED_RESULTS -->'),1)
                self.assertIn('Exact key coverage: False',report)
                self.assertTrue((root/'MEASURED_RESULTS.md').exists())
                finalizer.run()
                self.assertTrue((root/'failure_strata.png').is_file())
                self.assertFalse(json.loads((root/'finalization_status.json').read_text())['baseline_complete'])
                self.assertIn('incomplete/invalid-prediction diagnostic',(root/'FAILURE_ANALYSIS.md').read_text())
            finally:supervisor.OUT=previous_out
            print('Synthetic artifacts:',root)
        finally:a.OUT,a.DATA,a.ROOT,a.PRED,a.IDS=old

if __name__=='__main__':unittest.main()
