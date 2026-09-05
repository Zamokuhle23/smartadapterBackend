from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from apps.accounts.api import LearnerProfileView, RegisterView
from apps.progress.api import DashboardView, RecordAttemptView, WorkspaceView
from apps.quiz.api import (
    AnswerQuizView,
    CropAnswerView,
    ExamNextView,
    ExamStateView,
    GenerateQuizView,
    NextCropView,
    NextQuestionView,
    PaperAnchorsView,
    StartExamView,
)
from apps.syllabus.api import (
    DocumentUploadViewSet,
    EnrollmentViewSet,
    SubjectViewSet,
    SyllabusViewSet,
    TopicViewSet,
)
from apps.tutoring.api import ChatSessionViewSet


class _ThrottledTokenView(TokenObtainPairView):
    """Login endpoint fortified with a global brute-force throttle scope."""

    throttle_scope = "auth"


router = DefaultRouter()
router.register("syllabi", SyllabusViewSet, basename="syllabus")
router.register("subjects", SubjectViewSet, basename="subject")
router.register("topics", TopicViewSet, basename="topic")
router.register("documents", DocumentUploadViewSet, basename="document")
router.register("my-subjects", EnrollmentViewSet, basename="enrollment")
router.register("chat-sessions", ChatSessionViewSet, basename="chat-session")

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/auth/register/", RegisterView.as_view(), name="auth-register"),
    path("api/auth/token/", _ThrottledTokenView.as_view(), name="token_obtain_pair"),
    path("api/auth/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("api/me/profile/", LearnerProfileView.as_view(), name="learner-profile"),
    path("api/progress/attempt/", RecordAttemptView.as_view(), name="record-attempt"),
    path("api/progress/dashboard/", DashboardView.as_view(), name="progress-dashboard"),
    path("api/workspace/<int:subject_id>/", WorkspaceView.as_view(), name="workspace"),
    path("api/quiz/generate/", GenerateQuizView.as_view(), name="quiz-generate"),
    path("api/quiz/next/", NextQuestionView.as_view(), name="quiz-next"),
    path("api/quiz/answer/", AnswerQuizView.as_view(), name="quiz-answer"),
    path("api/quiz/crop/next/", NextCropView.as_view(), name="crop-next"),
    path("api/quiz/crop/answer/", CropAnswerView.as_view(), name="crop-answer"),
    path("api/quiz/paper/<int:doc_id>/anchors/", PaperAnchorsView.as_view(),
         name="paper-anchors"),
    path("api/quiz/exam/start/", StartExamView.as_view(), name="exam-start"),
    path("api/quiz/exam/<int:pk>/", ExamStateView.as_view(), name="exam-state"),
    path("api/quiz/exam/<int:pk>/next/", ExamNextView.as_view(), name="exam-next"),
    path("api/", include(router.urls)),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

