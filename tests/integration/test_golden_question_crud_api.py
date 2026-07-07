from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect
from app.main import create_app

pytestmark = pytest.mark.integration


def _seed_user_id(database_url: str) -> int:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT user_id FROM app_users WHERE login_id = 'alice.member'")
            return int(cursor.fetchone()["user_id"])


def _cleanup_question_set(database_url: str, set_name: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM golden_question_sets WHERE set_name = %s", (set_name,))


def test_golden_question_crud_api_lifecycle(migrated_database_url: str) -> None:
    user_id = _seed_user_id(migrated_database_url)
    set_name = f"slice-041-{uuid4()}"
    imported_set_name = f"{set_name}-imported"
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            create_set_response = client.post(
                "/api/evaluations/question-sets",
                json={
                    "set_name": f" {set_name} ",
                    "description": "Slice 041 fixture",
                    "metadata": {"slice": "041"},
                    "created_by_user_id": user_id,
                },
            )
            created_set = create_set_response.json()["question_set"]
            question_set_id = created_set["question_set_id"]

            get_set_response = client.get(f"/api/evaluations/question-sets/{question_set_id}")
            update_set_response = client.put(
                f"/api/evaluations/question-sets/{question_set_id}",
                json={
                    "set_name": set_name,
                    "description": "Slice 041 updated fixture",
                    "is_active": False,
                    "metadata": {"updated": True},
                    "created_by_user_id": user_id,
                },
            )
            list_sets_response = client.get(
                "/api/evaluations/question-sets",
                params={"active_only": False},
            )
            create_question_response = client.post(
                "/api/evaluations/questions",
                json={
                    "question_set_id": question_set_id,
                    "question_text": " What does Slice 041 cover? ",
                    "question_type": "single_fact",
                    "actor_user_id": user_id,
                    "requested_search_scope": "company",
                    "document_group": "slice-041",
                    "file_type": ".md",
                    "chunk_policy_name": "heading_512_64",
                    "top_k": 5,
                    "metadata": {"difficulty": "easy"},
                    "created_by_user_id": user_id,
                },
            )
            created_question = create_question_response.json()["question"]
            question_id = created_question["question_id"]

            list_questions_response = client.get(
                f"/api/evaluations/question-sets/{question_set_id}/questions",
                params={"requested_search_scope": "company"},
            )
            get_question_response = client.get(f"/api/evaluations/questions/{question_id}")
            update_question_response = client.put(
                f"/api/evaluations/questions/{question_id}",
                json={
                    "question_set_id": question_set_id,
                    "question_text": "Which updated Slice 041 behavior is covered?",
                    "question_type": "section",
                    "actor_user_id": user_id,
                    "requested_search_scope": "team",
                    "document_group": "slice-041",
                    "file_type": ".md",
                    "chunk_policy_name": "heading_512_64",
                    "top_k": 3,
                    "metadata": {"difficulty": "medium"},
                    "created_by_user_id": user_id,
                },
            )
            create_target_response = client.post(
                "/api/evaluations/expected-targets",
                json={
                    "question_id": question_id,
                    "expected_heading_path": ["Policy", "Scope"],
                    "expectation_type": "hidden",
                    "relevance_grade": 0,
                    "notes": "permission boundary",
                    "metadata": {"case": "hidden"},
                },
            )
            created_target = create_target_response.json()["expected_target"]
            expected_target_id = created_target["expected_target_id"]

            list_targets_response = client.get(
                f"/api/evaluations/questions/{question_id}/expected-targets"
            )
            get_target_response = client.get(
                f"/api/evaluations/expected-targets/{expected_target_id}"
            )
            update_target_response = client.put(
                f"/api/evaluations/expected-targets/{expected_target_id}",
                json={
                    "question_id": question_id,
                    "expected_heading_path": ["Policy", "Updated"],
                    "expectation_type": "visible",
                    "relevance_grade": 2,
                    "notes": "updated expectation",
                    "metadata": {"case": "visible"},
                },
            )
            export_response = client.get(f"/api/evaluations/question-sets/{question_set_id}/export")
            exported_payload = export_response.json()
            exported_payload["question_set"]["set_name"] = imported_set_name
            import_response = client.post(
                "/api/evaluations/question-sets/import",
                json=exported_payload,
            )
            imported = import_response.json()["imported"]
            bad_import_response = client.post(
                "/api/evaluations/question-sets/import",
                json={**exported_payload, "version": 999},
            )
            delete_target_response = client.delete(
                f"/api/evaluations/expected-targets/{expected_target_id}"
            )
            missing_target_response = client.get(
                f"/api/evaluations/expected-targets/{expected_target_id}"
            )
            delete_question_response = client.delete(f"/api/evaluations/questions/{question_id}")
            missing_question_response = client.get(f"/api/evaluations/questions/{question_id}")
            delete_set_response = client.delete(f"/api/evaluations/question-sets/{question_set_id}")
            missing_set_response = client.get(f"/api/evaluations/question-sets/{question_set_id}")
            bad_set_response = client.post(
                "/api/evaluations/question-sets",
                json={"set_name": " "},
            )

        assert create_set_response.status_code == 201
        assert created_set["set_name"] == set_name
        assert created_set["metadata"] == {"slice": "041"}
        assert get_set_response.status_code == 200
        assert get_set_response.json()["question_set"]["question_set_id"] == question_set_id
        assert update_set_response.status_code == 200
        assert update_set_response.json()["question_set"]["is_active"] is False
        assert update_set_response.json()["question_set"]["metadata"] == {"updated": True}
        assert question_set_id in {
            item["question_set_id"] for item in list_sets_response.json()["question_sets"]
        }

        assert create_question_response.status_code == 201
        assert created_question["question_text"] == "What does Slice 041 cover?"
        assert created_question["normalized_question_text"] == "what does slice 041 cover?"
        assert list_questions_response.status_code == 200
        assert [item["question_id"] for item in list_questions_response.json()["questions"]] == [
            question_id
        ]
        assert get_question_response.status_code == 200
        assert get_question_response.json()["question"]["question_id"] == question_id
        assert get_question_response.json()["expected_targets"] == []
        assert update_question_response.status_code == 200
        assert update_question_response.json()["question"]["question_type"] == "section"
        assert update_question_response.json()["question"]["top_k"] == 3

        assert create_target_response.status_code == 201
        assert created_target["expected_heading_path"] == ["Policy", "Scope"]
        assert created_target["expectation_type"] == "hidden"
        assert list_targets_response.status_code == 200
        assert list_targets_response.json()["expected_targets"][0]["expected_target_id"] == (
            expected_target_id
        )
        assert get_target_response.status_code == 200
        assert get_target_response.json()["expected_target"]["notes"] == "permission boundary"
        assert update_target_response.status_code == 200
        assert update_target_response.json()["expected_target"]["expectation_type"] == "visible"
        assert update_target_response.json()["expected_target"]["relevance_grade"] == 2
        assert export_response.status_code == 200
        assert exported_payload["version"] == 1
        assert exported_payload["question_set"]["set_name"] == imported_set_name
        assert exported_payload["questions"][0]["question_text"] == (
            "Which updated Slice 041 behavior is covered?"
        )
        assert exported_payload["questions"][0]["expected_targets"][0]["expectation_type"] == (
            "visible"
        )
        assert import_response.status_code == 201
        assert imported["question_set"]["set_name"] == imported_set_name
        assert imported["questions"][0]["question_text"] == (
            "Which updated Slice 041 behavior is covered?"
        )
        assert imported["expected_targets"][0]["expected_heading_path"] == ["Policy", "Updated"]
        assert bad_import_response.status_code == 400
        assert delete_target_response.status_code == 204
        assert missing_target_response.status_code == 404
        assert delete_question_response.status_code == 204
        assert missing_question_response.status_code == 404
        assert delete_set_response.status_code == 204
        assert missing_set_response.status_code == 404
        assert bad_set_response.status_code == 400
    finally:
        _cleanup_question_set(migrated_database_url, set_name)
        _cleanup_question_set(migrated_database_url, imported_set_name)
