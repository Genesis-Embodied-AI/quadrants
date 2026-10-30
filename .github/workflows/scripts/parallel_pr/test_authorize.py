"""Offline tests of the comment command's authorization boundary."""

import unittest
from unittest.mock import patch

import authorize


class AuthorizationTests(unittest.TestCase):
    def event(self):
        return {
            "action": "created",
            "issue": {"pull_request": {"url": "https://api.github.com/repos/org/repo/pulls/1"}},
            "comment": {"body": "/make_parallel", "user": {"login": "maintainer", "type": "User"}},
            "repository": {"full_name": "org/repo"},
        }

    def test_current_permission_controls_access(self):
        for permission, allowed in (("write", True), ("admin", True), ("read", False), ("none", False)):
            with (
                self.subTest(permission=permission),
                patch.object(authorize, "api", return_value={"permission": permission}) as api,
            ):
                self.assertEqual(authorize.authorized(self.event()), allowed)
                api.assert_called_once_with("repos/org/repo/collaborators/maintainer/permission")

    def test_ignores_other_events_and_commands_without_api_access(self):
        events = []
        for body in ("please /make_parallel", "/make_parallel; echo bad", "/make_parallel extra", "hello"):
            event = self.event()
            event["comment"]["body"] = body
            events.append(event)
        event = self.event()
        event["action"] = "edited"
        events.append(event)
        event = self.event()
        event["issue"] = {}
        events.append(event)
        event = self.event()
        event["comment"]["user"]["type"] = "Bot"
        events.append(event)
        with patch.object(authorize, "api") as api:
            for event in events:
                self.assertFalse(authorize.authorized(event))
            api.assert_not_called()

    def test_whitespace_is_allowed(self):
        event = self.event()
        event["comment"]["body"] = " /make_parallel\n"
        with patch.object(authorize, "api", return_value={"permission": "write"}):
            self.assertTrue(authorize.authorized(event))

    def test_lookup_failure_does_not_authorize(self):
        with patch.object(authorize, "api", side_effect=RuntimeError("lookup failed")):
            with self.assertRaisesRegex(RuntimeError, "lookup failed"):
                authorize.authorized(self.event())
