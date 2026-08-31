from celery import shared_task


@shared_task(bind=True, max_retries=2, default_retry_delay=30)
def ingest_document_task(self, document_id: int) -> int:
    from apps.syllabus.models import SyllabusDocument
    from apps.syllabus.services.ingestion import process_document

    document = SyllabusDocument.objects.get(pk=document_id)
    try:
        return process_document(document)
    except Exception as exc:  # noqa: BLE001
        raise self.retry(exc=exc)
