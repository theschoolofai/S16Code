"""No capability may hand the model a path to the machine it runs on.

The model quotes what it is shown. `read_file` returned the resolved absolute
path, and `read_file`'s evidence projection promotes that key into the
`[source: ...]` annotation the answer worker reads, so a Telegram reply to the
owner arrived containing the operator's home directory and Windows username.

The gateway now also redacts host paths at the outbound envelope, but that is a
net. This pins the producer: the path must not exist in the model's context in
the first place, because text that is never produced cannot leak through a
surface nobody thought to net.
"""

from __future__ import annotations

import re

from s16code.capabilities import default_registry, project_evidence

# A drive letter, a UNC prefix, or a POSIX home directory.
#
# The word boundary is load-bearing: without it `[A-Za-z]:[\\/]` matches the
# "x:/" inside "sandbox://reminders.txt" and the guard fires on the very shape
# it is meant to bless. A drive letter is one letter with nothing before it.
HOST_PATH = re.compile(r"(?i)(\b[A-Za-z]:[\\/]|\\\\|/home/|/Users/)")


class TestReadFileEvidence:
    def _projection(self):
        capability = default_registry().get("read_file")
        assert capability.evidence is not None
        return capability.evidence

    def test_the_source_names_the_file_not_the_machine(self) -> None:
        records = project_evidence(
            self._projection(),
            {"path": "reminders.txt", "text": "MY REMINDERS AND TO-DOS"},
            fallback_source="graph://run/test",
        )
        sources = [source for record in records for source in record.get("sources", [])]
        assert sources == ["sandbox://reminders.txt"]

    def test_a_worker_that_regresses_is_caught_by_the_projector(self) -> None:
        # Defence in depth. The worker is fixed, but if a future change makes it
        # return an absolute path again, the projector must refuse to promote it
        # rather than render it into the annotation the answer worker reads.
        records = project_evidence(
            self._projection(),
            {"path": r"C:\Users\alice\sandbox\reminders.txt", "text": "..."},
            fallback_source="graph://run/test",
        )
        sources = [source for record in records for source in record.get("sources", [])]
        assert not [source for source in sources if HOST_PATH.search(source)], sources
        assert sources == ["graph://run/test"], \
            "a rejected source must fall back to internal provenance, not vanish"

    def test_no_evidence_text_or_source_carries_a_host_path(self) -> None:
        records = project_evidence(
            self._projection(),
            {"path": "reminders.txt", "text": "Dentist appointment, 6:30pm, Dr Rao."},
            fallback_source="graph://run/test",
        )
        for record in records:
            assert not HOST_PATH.search(str(record.get("text", "")))
            for source in record.get("sources", []):
                assert not HOST_PATH.search(source), source


class TestEveryCapabilityDeclaration:
    """A future capability must not reintroduce the same template."""

    def test_no_source_template_builds_a_file_uri(self) -> None:
        offenders = []
        for name in default_registry().names():
            capability = default_registry().get(name)
            projection = getattr(capability, "evidence", None)
            template = getattr(projection, "source_template", None) if projection else None
            if template and template.lower().startswith("file:"):
                offenders.append(f"{name}: {template}")
        assert offenders == [], (
            "a file:// source template renders a host path into the model's "
            f"context; use sandbox:// or artifact://. Offenders: {offenders}")
