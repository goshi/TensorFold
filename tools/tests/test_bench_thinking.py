"""Model-free discovery and truthful thinking-mode requests for both benchmark clients."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_concurrent
import bench_openai
from test_bench_auth import response


class ThinkingDiscovery(unittest.TestCase):
    def test_stream_omits_an_unsupported_switch(self):
        for module in (bench_openai, bench_concurrent):
            item = {**module.PROMPTS[1], "thinking_switch": False}
            with self.subTest(client=module.__name__), patch.object(module.urllib.request, "urlopen", return_value=response()) as http:
                module.stream("http://unused", "fixture", item, 4, 0, 1)
                self.assertNotIn("chat_template_kwargs", json.loads(http.call_args.args[0].data))

    def run_cli(self, module, bit, *, document=None):
        seen = []
        def opened(req, **kwargs):
            seen.append(req)
            if req.get_method() == "GET":
                doc = document if document is not None else {"data": [{"id": "fixture", "capabilities": {"thinking_switch": bit}}]}
                return io.BytesIO(json.dumps(doc).encode())
            return response()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "receipt.json"
            argv = [module.__name__, "http://unused", "fixture", "--tokens", "4", "--reps", "1", "--temperatures", "0", "--output", str(out)]
            if module is bench_concurrent:
                argv += ["--levels", "1,2", "--alone", "--serial"]
            with patch.object(sys, "argv", argv), patch.object(module.urllib.request, "urlopen", side_effect=opened), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            receipt = json.loads(out.read_text())
        return seen, receipt

    def test_one_discovery_controls_every_chat_and_receipt(self):
        for module in (bench_openai, bench_concurrent):
            for bit in (True, False, None, "false", 1):
                with self.subTest(client=module.__name__, bit=bit):
                    seen, receipt = self.run_cli(module, bit)
                    get = [r for r in seen if r.get_method() == "GET"]
                    self.assertEqual(len(get), 1)
                    self.assertEqual(get[0].full_url, "http://unused/v1/models")
                    for req in seen:
                        if req.data is None:
                            continue
                        body = json.loads(req.data)
                        if "messages" in body and bit is True:
                            self.assertIs(body["chat_template_kwargs"]["enable_thinking"], False)
                        else:
                            self.assertNotIn("chat_template_kwargs", body)
                    rows = receipt if module is bench_openai else receipt["cells"]
                    chats = [r for r in rows if r["prompt"] in ("chat", "gpu-chat", "gpu-chat-no-think")]
                    self.assertTrue(chats)
                    for row in chats:
                        self.assertEqual(row["thinking_off_requested"], bit is True)
                        self.assertEqual(row["thinking"], "off" if bit is True else "default (thinking off unsupported)" if bit is False else "default (thinking switch unknown)")

    def test_another_models_bit_and_missing_capability_are_unknown(self):
        for module in (bench_openai, bench_concurrent):
            for doc in ({"data": [{"id": "other", "capabilities": {"thinking_switch": True}}]}, {"data": [{"id": "fixture"}]}):
                seen, receipt = self.run_cli(module, None, document=doc)
                for req in seen:
                    if req.data:
                        self.assertNotIn("chat_template_kwargs", json.loads(req.data))


if __name__ == "__main__":
    unittest.main()
