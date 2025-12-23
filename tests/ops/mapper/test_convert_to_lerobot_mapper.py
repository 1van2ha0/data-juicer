import unittest
from unittest.mock import MagicMock, patch
import os
import yaml
import h5py
import numpy as np
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class ConvertToLeRobotMapperTest(DataJuicerTestCaseBase):
    
    def setUp(self):
        super().setUp()
        self.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp_convert')
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.input_path = os.path.join(self.tmp_dir, 'input.hdf5')
        self.output_dir = os.path.join(self.tmp_dir, 'output')
        self.config_path = os.path.join(self.tmp_dir, 'config.yaml')
        self.modality_path = os.path.join(self.tmp_dir, 'modality.json')
        self.info_path = os.path.join(self.tmp_dir, 'info.json')
        
        # Create dummy HDF5
        with h5py.File(self.input_path, 'w') as f:
            data = f.create_group('data')
            demo = data.create_group('demo_0')
            demo.create_dataset('obs/state', data=np.random.rand(10, 7))
            demo.create_dataset('actions', data=np.random.rand(10, 7))
            demo.attrs['num_samples'] = 10
            
        # Create dummy config
        config = {
            'dataset': {'robot_type': 'franka', 'fps': 10},
            'modality_template_path': self.modality_path,
            'info_template_path': self.info_path,
            'tasks': {'0': 'test task'},
            'external_config': {
                'mapping': {
                    'state': [{'key': 'obs/state'}],
                    'action': [{'key': 'actions'}]
                },
                'lerobot_keys': {
                    'state': 'observation.state',
                    'action': 'action'
                }
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
            
        with open(self.modality_path, 'w') as f:
            f.write('{}')
        with open(self.info_path, 'w') as f:
            f.write('{}')

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            import shutil
            shutil.rmtree(self.tmp_dir)

    @patch('subprocess.check_output')
    def test_process_batched(self, mock_subprocess):
        # Mock ffprobe output
        mock_subprocess.return_value = b'{"streams": [{"height": 100, "width": 100, "codec_name": "h264", "pix_fmt": "yuv420p", "r_frame_rate": "30/1"}]}'
        
        from data_juicer.ops.mapper.convert_to_lerobot_mapper import ConvertToLeRobotMapper
        
        op = ConvertToLeRobotMapper(config_path=self.config_path)
        
        samples = {
            'text': ['dummy'],
            'input_file': [self.input_path],
            'output_dir': [self.output_dir],
            'video_dir': [None],
            'config_path': [None]
        }
        
        op.process_batched(samples)
        
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'meta/tasks.jsonl')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'meta/episodes.jsonl')))

if __name__ == '__main__':
    unittest.main()
