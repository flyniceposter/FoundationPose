"""Fake-estimator test: checkpoint/resume without importing GPU libraries."""
import sys,tempfile,unittest,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,yaml
import checkpoint_runner as r
from recover_log import parse
class CheckpointTests(unittest.TestCase):
    def test_log_mapping_and_truncation(self):
        text='[run_pose_estimation_worker()] 0/1, i_frame:1, ob_id:2\n[run_pose_estimation_worker()] pose:\n[[1 0 0 0]\n [0 1 0 0]\n [0 0 1 1]\n [0 0 0 1]]\n'
        poses,records=parse(text,{2:[7,19]});self.assertIn((2,19,2),poses)
        with self.assertRaises(ValueError):parse(text[:-8],{2:[7,19]})
    def test_resume_after_crash(self):
        root=Path(tempfile.mkdtemp(prefix='foundationpose-checkpoint-test-'));oldroot=r.ROOT;oldexpected=r.expected;oldconflict=r.gpu_conflict;prior=sys.modules.get('run_linemod')
        calls=[];crash=[True];fake=SimpleNamespace(set_seed=lambda n:None)
        reader=SimpleNamespace(id_strs=['000000','000001'],get_video_id=lambda:1)
        def actual(reader,frames,*args,**kwargs):
            im=frames[0]
            if im==1 and crash[0]:raise RuntimeError('Simulated interruption')
            calls.append(im);p=np.eye(4,dtype=np.float32);p[2,3]=1
            return {1:{reader.id_strs[im]:{1:p}}}
        fake.run_pose_estimation_worker=actual
        def upstream_run():
            result={1:{}}
            for im in range(2):result[1].update(fake.run_pose_estimation_worker(reader,[im],None,0,1,'cuda:0')[1])
            dump={1:{im:{1:p[1].tolist()} for im,p in result[1].items()}}
            (Path(fake.opt.debug_dir)/'linemod_res.yml').write_text(yaml.safe_dump(dump))
        fake.run_pose_estimation=upstream_run
        try:
            r.ROOT=root;r.expected=lambda:[(1,0,1),(1,1,1)];r.gpu_conflict=lambda:None
            (root/'research').mkdir();(root/'weights').mkdir()
            for name in ['run_linemod.py','estimater.py','datareader.py','Utils.py']:(root/name).write_text('fixture')
            sys.modules['run_linemod']=fake;args=SimpleNamespace(output=root/'output',reuse_recovered=None,run=True)
            with self.assertRaisesRegex(RuntimeError,'Simulated interruption'):r.run(args)
            self.assertTrue((root/'output/checkpoints/000001_000000_000001.npy').exists())
            self.assertFalse((root/'output/completion.json').exists())
            crash[0]=False;fake.run_pose_estimation_worker=actual;r.run(args)
            self.assertEqual(calls,[0,1]);self.assertTrue(json.loads((root/'output/completion.json').read_text())['inference_returned_normally'])
            with self.assertRaisesRegex(RuntimeError,'already completed'):r.run(args)
        finally:
            r.ROOT=oldroot;r.expected=oldexpected;r.gpu_conflict=oldconflict
            if prior is None:sys.modules.pop('run_linemod',None)
            else:sys.modules['run_linemod']=prior
if __name__=='__main__':unittest.main()
