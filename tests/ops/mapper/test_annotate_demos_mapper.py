import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import numpy as np
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

@patch.dict(sys.modules, {
    'isaaclab': MagicMock(),
    'isaaclab.app': MagicMock(),
    'isaaclab.envs.mdp.recorders.recorders_cfg': MagicMock(),
    'isaaclab.managers': MagicMock(),
    'isaaclab.utils': MagicMock(),
    'isaaclab.utils.configclass': MagicMock(),
    'isaaclab_tasks.utils.parse_cfg': MagicMock(),
    'isaaclab.utils.datasets': MagicMock(),
    'gymnasium': MagicMock(),
    'torch': MagicMock(),
})
class AnnotateDemosMapperTest(DataJuicerTestCaseBase):
    
    def setUp(self):
        super().setUp()
        self.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp_annotate')
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.input_path = os.path.join(self.tmp_dir, 'input.hdf5')
        self.output_path = os.path.join(self.tmp_dir, 'output.hdf5')
        # Create dummy input file
        with open(self.input_path, 'wb') as f:
            f.write(b'dummy hdf5 content')

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            import shutil
            shutil.rmtree(self.tmp_dir)

    def test_process_batched(self):
        mock_torch = sys.modules['torch']
        # Mock torch cuda
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        
        # Mock isaaclab components
        mock_isaac_utils = sys.modules['isaaclab.utils.datasets']
        mock_handler = MagicMock()
        mock_isaac_utils.HDF5DatasetFileHandler.return_value = mock_handler
        mock_handler.get_num_episodes.return_value = 1
        mock_handler.get_episode_names.return_value = ['episode_0']
        
        # Mock episode data
        mock_episode = MagicMock()
        # Create dummy actions: 5 steps, 7 dims
        actions = np.zeros((5, 7))
        initial_state = np.zeros(10)
        mock_episode.data = {
            "initial_state": initial_state,
            "actions": actions
        }
        mock_handler.load_episode.return_value = mock_episode

        # Mock gym environment
        mock_gym = sys.modules['gymnasium']
        mock_env = MagicMock()
        mock_gym.make.return_value.unwrapped = mock_env
        
        # Mock recorder manager output
        # This simulates the annotated data returned by Isaac Lab
        mock_annotated_episode = MagicMock()
        mock_annotated_episode.data = {
            "obs": {
                "datagen_info": {
                    "subtask_term_signals": {
                        "term_1": [0, 0, 1, 0, 0]  # Simulated signal
                    }
                }
            }
        }
        mock_env.recorder_manager.get_episode.return_value = mock_annotated_episode

        from data_juicer.ops.mapper.annotate_demos_mapper import AnnotateDemosMapper
        
        op = AnnotateDemosMapper(task_name="Test-Task", headless=True)
        
        # Mock _ensure_sim_app to avoid real app launch
        op._ensure_sim_app = MagicMock()
        
        # Mock _create_task_env to return our mock_env and avoid importing real config classes
        # We want to test _replay_and_annotate logic, so we let it run
        def create_env_side_effect():
            op._env = mock_env
            return mock_env
        op._create_task_env = MagicMock(side_effect=create_env_side_effect)
        
        samples = {
            'text': ['dummy'],
            'input_file': [self.input_path],
            'output_file': [self.output_path]
        }
        
        res = op.process_batched(samples)
        
        self.assertEqual(len(res['input_file']), 1)
        op._ensure_sim_app.assert_called()
        mock_handler.open.assert_called_with(self.input_path)
        
        # Verify Replay Logic
        # 1. Verify reset was called
        mock_env.reset_to.assert_called_with(initial_state, None, is_relative=True)
        
        # 2. Verify step was called for each action (5 times)
        self.assertEqual(mock_env.step.call_count, 5)
        
        # 3. Verify recorder manager was queried
        mock_env.recorder_manager.get_episode.assert_called_with(0)


if __name__ == '__main__':
    unittest.main()
