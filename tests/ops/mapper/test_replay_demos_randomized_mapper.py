import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import h5py
import numpy as np
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

@patch.dict(sys.modules, {
    'isaaclab': MagicMock(),
    'isaaclab.app': MagicMock(),
    'isaaclab_tasks': MagicMock(),
    'isaaclab_tasks.utils': MagicMock(),
    'isaaclab_tasks.utils.parse_cfg': MagicMock(),
    'isaaclab.managers': MagicMock(),
    'isaaclab.utils.assets': MagicMock(),
    'isaaclab.utils.datasets': MagicMock(),
    'gymnasium': MagicMock(),
    'torch': MagicMock(),
    'cv2': MagicMock(),
})
class ReplayDemosRandomizedMapperTest(DataJuicerTestCaseBase):
    
    def setUp(self):
        super().setUp()
        self.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp_replay')
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.input_path = os.path.join(self.tmp_dir, 'input.hdf5')
        self.output_path = os.path.join(self.tmp_dir, 'output.hdf5')
        self.video_dir = os.path.join(self.tmp_dir, 'videos')
        
        # Create dummy HDF5
        with h5py.File(self.input_path, 'w') as f:
            data = f.create_group('data')
            demo = data.create_group('demo_0')
            demo.create_dataset('initial_state', data=np.zeros(10))
            demo.create_dataset('actions', data=np.zeros((10, 7)))
            demo.attrs['num_samples'] = 10

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            import shutil
            shutil.rmtree(self.tmp_dir)

    def test_process_batched(self):
        mock_torch = sys.modules['torch']
        mock_torch.cuda.is_available.return_value = True
        
        # Define a dummy class for torch.Tensor so isinstance checks work
        class MockTensorType:
            def reshape(self, *args, **kwargs): pass
            def cpu(self): pass
            def numpy(self): pass
            def __getitem__(self, item): pass
            ndim = 1
            shape = (1, 7)

        # Create a mock tensor instance that passes isinstance(x, MockTensorType)
        mock_tensor = MagicMock(spec=MockTensorType)
        mock_tensor.ndim = 1
        mock_tensor.shape = (1, 7)
        mock_tensor.reshape.return_value = mock_tensor
        mock_tensor.cpu.return_value = mock_tensor
        mock_tensor.numpy.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_tensor.__getitem__.return_value = 0.0
        
        # Set torch.Tensor to be the type itself
        mock_torch.Tensor = MockTensorType
        # Ensure torch.tensor() returns our mock instance
        mock_torch.tensor.side_effect = lambda data, **kwargs: mock_tensor
        
        # Mock isaaclab datasets
        mock_isaac_utils = sys.modules['isaaclab.utils.datasets']
        mock_handler = MagicMock()
        mock_isaac_utils.HDF5DatasetFileHandler.return_value = mock_handler
        mock_handler.get_episode_names.return_value = ['demo_0']
        mock_episode = MagicMock()
        # Ensure both initial_state and actions are present in data
        mock_episode.data = {
            'initial_state': MagicMock(),
            'actions': MagicMock() # Mock actions tensor/array
        }
        mock_episode.get_next_action.side_effect = [np.zeros(7), None]
        
        # Mock actions to be iterable (length 1)
        mock_episode.data['actions'].__len__.return_value = 1
        mock_episode.data['actions'].__getitem__.return_value = np.zeros(7)
        
        mock_handler.load_episode.return_value = mock_episode
        
        # Mock gym
        mock_gym = sys.modules['gymnasium']
        mock_env = MagicMock()
        mock_env.reset.return_value = (MagicMock(), {})
        mock_env.step.return_value = (MagicMock(), 0.0, False, False, {})
        mock_gym.make.return_value.unwrapped = mock_env
        
        # Mock parse_env_cfg to return an object we can inspect for randomization injection
        mock_env_cfg = MagicMock()
        mock_env_cfg.events = MagicMock() # Ensure events attribute exists
        # Ensure other attributes checked by _create_env exist to avoid AttributeErrors
        mock_env_cfg.terminations = MagicMock()
        mock_env_cfg.observations = MagicMock()
        mock_env_cfg.sim = MagicMock()
        
        mock_isaac_tasks_utils = sys.modules['isaaclab_tasks.utils.parse_cfg']
        mock_isaac_tasks_utils.parse_env_cfg.return_value = mock_env_cfg

        # Create dummy visual randomization config
        config_path = os.path.join(self.tmp_dir, 'rand.yaml')
        with open(config_path, 'w') as f:
            f.write('test_event:\n  func: test_func\n  params: {}\n')

        from data_juicer.ops.mapper.replay_demos_randomized_mapper import ReplayDemosRandomizedMapper
        
        # Initialize with visual randomization config
        op = ReplayDemosRandomizedMapper(
            task_name="Test-Task", 
            headless=True, 
            video=True, 
            camera_view_list=['front'],
            visual_randomization_config=config_path
        )
        op._ensure_sim_app = MagicMock()
        # Mock _inject_visual_randomization to avoid complex config resolution issues in test
        op._inject_visual_randomization = MagicMock()
        
        # Mock subprocess to verify ffmpeg call
        with patch('subprocess.run') as mock_run:
            samples = {
                'text': ['dummy'],
                'dataset_file': [self.input_path],
                'output_file': [self.output_path],
                'video_dir': [self.video_dir]
            }
            
            res = op.process_batched(samples)
            
            self.assertTrue(res['replay_success'][0])
            op._ensure_sim_app.assert_called()
            
            # Verify ffmpeg was called
            self.assertTrue(mock_run.called)
            args, _ = mock_run.call_args
            cmd = args[0]
            self.assertIn('ffmpeg', cmd)
            self.assertIn(str(20.0), cmd) # Default framerate
            
            # Verify visual randomization was injected (called)
            op._inject_visual_randomization.assert_called()


if __name__ == '__main__':
    unittest.main()
