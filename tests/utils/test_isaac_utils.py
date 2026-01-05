import unittest
from unittest.mock import MagicMock, patch, mock_open
import sys
import os
from typing import Dict, List, Union, cast
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.isaac_utils import (
    LazyStreamRedirector,
    init_isaac_sim_app,
    ensure_isaac_sim_app,
    cleanup_isaac_env,
    resolve_nucleus_paths,
    create_video_from_images
)

class TestIsaacUtils(DataJuicerTestCaseBase):

    def test_lazy_stream_redirector(self):
        # Mock sys streams
        with patch('sys.stdin', new_callable=MagicMock) as mock_stdin, \
             patch('sys.stdout', new_callable=MagicMock) as mock_stdout, \
             patch('sys.stderr', new_callable=MagicMock) as mock_stderr:
            
            # Set up real streams mocks
            sys.__stdin__ = MagicMock()
            sys.__stdout__ = MagicMock()
            sys.__stderr__ = MagicMock()
            
            with LazyStreamRedirector():
                self.assertEqual(sys.stdin, sys.__stdin__)
                self.assertEqual(sys.stdout, sys.__stdout__)
                self.assertEqual(sys.stderr, sys.__stderr__)
            
            # Should be restored
            self.assertEqual(sys.stdin, mock_stdin)
            self.assertEqual(sys.stdout, mock_stdout)
            self.assertEqual(sys.stderr, mock_stderr)
            
            # Clean up
            del sys.__stdin__
            del sys.__stdout__
            del sys.__stderr__

    @patch('data_juicer.utils.isaac_utils.torch')
    @patch('data_juicer.utils.isaac_utils.logger')
    def test_init_isaac_sim_app(self, mock_logger, mock_torch):
        # Mock torch cuda availability
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        
        # Mock isaaclab.app.AppLauncher
        with patch.dict(sys.modules, {'isaaclab.app': MagicMock()}):
            # We need to mock the import inside the function or ensure sys.modules is checked
            # Since the function does `from isaaclab.app import AppLauncher`, 
            # and we patched sys.modules, it should work.
            
            # We also need to mock AppLauncher class on the mocked module
            mock_module = sys.modules['isaaclab.app']
            mock_launcher_class = MagicMock()
            mock_module.AppLauncher = mock_launcher_class
            
            mock_launcher_instance = MagicMock()
            mock_launcher_class.return_value = mock_launcher_instance
            
            # Mock faulthandler
            with patch('data_juicer.utils.isaac_utils.faulthandler') as mock_faulthandler:
                app = init_isaac_sim_app(headless=True, device="cuda:0")
                
                self.assertTrue(mock_torch.cuda.empty_cache.called)
                self.assertTrue(mock_launcher_class.called)
                self.assertEqual(app, mock_launcher_instance.app)

    @patch('data_juicer.utils.isaac_utils.init_isaac_sim_app')
    def test_ensure_isaac_sim_app(self, mock_init):
        class MockMapper:
            def __init__(self):
                self.headless = True
                self.device = "cuda:0"
                self.enable_cameras = False
                self._isaac_initialized = False
                self._simulation_app = None
        
        mapper = MockMapper()
        
        # Mock imports that happen inside ensure_isaac_sim_app
        with patch.dict(sys.modules, {
            'isaaclab_mimic.envs': MagicMock(),
            'isaaclab_tasks': MagicMock()
        }):
            ensure_isaac_sim_app(mapper, mode='mimic')
            
            self.assertTrue(mock_init.called)
            self.assertTrue(mapper._isaac_initialized)
            self.assertIsNotNone(mapper._simulation_app)
            
            # Call again, should not init
            mock_init.reset_mock()
            ensure_isaac_sim_app(mapper, mode='mimic')
            self.assertFalse(mock_init.called)

    def test_cleanup_isaac_env(self):
        class MockMapper:
            def __init__(self):
                self._env = MagicMock()
                self._faulthandler_file = MagicMock()
        
        mapper = MockMapper()
        
        with patch('data_juicer.utils.isaac_utils.faulthandler') as mock_faulthandler:
            res = cleanup_isaac_env(mapper)
            
            self.assertTrue(mapper._env.close.called)
            self.assertIsNone(mapper._env)
            self.assertTrue(mock_faulthandler.disable.called)
            self.assertTrue(mapper._faulthandler_file.close.called)
            self.assertIsNone(mapper._faulthandler_file)
            self.assertEqual(res, {"status": "cleaned"})

    def test_resolve_nucleus_paths(self):
        # Mock isaaclab.utils.assets
        mock_assets = MagicMock()
        mock_assets.ISAAC_NUCLEUS_DIR = "/isaac/nucleus"
        mock_assets.NVIDIA_NUCLEUS_DIR = "/nvidia/nucleus"
        
        with patch.dict(sys.modules, {'isaaclab.utils.assets': mock_assets}):
            config = {
                "path1": "{ISAAC_NUCLEUS_DIR}/item",
                "path2": "{NVIDIA_NUCLEUS_DIR}/item",
                "nested": ["{ISAAC_NUCLEUS_DIR}/list"]
            }
            
            resolved = cast(Dict, resolve_nucleus_paths(config))
            
            self.assertIsInstance(resolved, dict)
            self.assertEqual(resolved["path1"], "/isaac/nucleus/item")
            self.assertEqual(resolved["path2"], "/nvidia/nucleus/item")
            self.assertEqual(resolved["nested"][0], "/isaac/nucleus/list")

    @patch('subprocess.run')
    def test_create_video_from_images(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        
        res = create_video_from_images("frame_%d.png", "out.mp4")
        
        self.assertTrue(res)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertIn("ffmpeg", args)
        self.assertIn("frame_%d.png", args)
        self.assertIn("out.mp4", args)

if __name__ == '__main__':
    unittest.main()
