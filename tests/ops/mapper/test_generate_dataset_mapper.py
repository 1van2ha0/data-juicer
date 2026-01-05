import unittest
from unittest.mock import MagicMock, patch
import os
import sys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

@patch.dict(sys.modules, {
    'isaaclab': MagicMock(),
    'isaaclab.app': MagicMock(),
    'isaaclab.envs': MagicMock(),
    'isaaclab_mimic': MagicMock(),
    'isaaclab_mimic.datagen.utils': MagicMock(),
    'isaaclab_mimic.datagen.generation': MagicMock(),
    'gymnasium': MagicMock(),
    'torch': MagicMock(),
    'omni': MagicMock(),
})
class GenerateDatasetMapperTest(DataJuicerTestCaseBase):
    
    def setUp(self):
        super().setUp()
        self.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp_generate')
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.input_path = os.path.join(self.tmp_dir, 'input.hdf5')
        self.output_path = os.path.join(self.tmp_dir, 'output.hdf5')

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            import shutil
            shutil.rmtree(self.tmp_dir)

    def test_process_batched(self):
        mock_torch = sys.modules['torch']
        mock_torch.cuda.is_available.return_value = True
        
        # Mock isaaclab utils
        mock_mimic_utils = sys.modules['isaaclab_mimic.datagen.utils']
        mock_mimic_utils.get_env_name_from_dataset.return_value = 'TestEnv'
        # Return real temp path to verify file existence check
        mock_mimic_utils.setup_output_paths.return_value = (self.tmp_dir, 'output.hdf5')
        
        # Mock generation
        mock_generation = sys.modules['isaaclab_mimic.datagen.generation']
        mock_generation.setup_env_config.return_value = (MagicMock(), MagicMock())
        mock_generation.setup_async_generation.return_value = {
            'tasks': [], 
            'reset_queue': MagicMock(), 
            'action_queue': MagicMock(), 
            'info_pool': MagicMock(), 
            'event_loop': MagicMock()
        }
        
        # Mock ManagerBasedRLMimicEnv for isinstance check
        class MockRLEnv:
            def close(self): pass
            def reset(self): pass
            cfg = MagicMock()
            def target_eef_pose_to_action(self): pass
        sys.modules['isaaclab.envs'].ManagerBasedRLMimicEnv = MockRLEnv
        
        mock_gym = sys.modules['gymnasium']
        mock_env = MagicMock(spec=MockRLEnv)
        # Fix inspect.signature check
        def dummy_action_method(action_noise_dict=None): pass
        mock_env.target_eef_pose_to_action = dummy_action_method
        # Fix seed check
        mock_env.cfg.datagen_config.seed = None
        mock_gym.make.return_value.unwrapped = mock_env
        
        from data_juicer.ops.mapper.generate_dataset_mapper import GenerateDatasetMapper
        
        op = GenerateDatasetMapper(task_name="Test-Task", headless=True, num_envs=8, generation_num_trials=1000)
        
        # Case 1: Success (File created)
        # Manually create the output file to simulate successful generation
        with open(self.output_path, 'w') as f:
            f.write('dummy content')

        samples = {
            'text': ['dummy'],
            'input_file': [self.input_path],
            'output_file': [self.output_path]
        }
        # Patch the function where it is imported in the module, or patch the module where it is defined
        # Since ensure_isaac_sim_app is imported inside the method _generate_dataset, we need to patch it in data_juicer.utils.isaac_utils
        with patch('data_juicer.utils.isaac_utils.ensure_isaac_sim_app') as mock_ensure_app:
            res = op.process_batched(samples)
            mock_ensure_app.assert_called()
            
        self.assertTrue(res['generation_result'][0]['success'])
        
        # Verify config passing
        mock_generation.setup_env_config.assert_called_with(
            env_name='Test-Task',
            output_dir=self.tmp_dir,
            output_file_name='output.hdf5',
            num_envs=8,
            device='cuda:auto',
            generation_num_trials=1000
        )

        # Case 2: Failure (File not created)
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
            
        with patch('data_juicer.utils.isaac_utils.ensure_isaac_sim_app') as mock_ensure_app:
            res = op.process_batched(samples)
            mock_ensure_app.assert_called()
            
        self.assertFalse(res['generation_result'][0]['success'])


if __name__ == '__main__':
    unittest.main()
