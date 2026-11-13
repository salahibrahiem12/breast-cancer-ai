import type { Metadata } from "next";
import { ContentCard, ContentPage } from "@/components/content-page";

export const metadata: Metadata = { title: "Privacy" };

export default function PrivacyPage() {
  return (
    <ContentPage
      eyebrow="Privacy"
      title="Uploads are not kept by the application."
      intro="This page describes the behavior implemented in the repository. It is not a claim of HIPAA compliance or a guarantee about the hosting provider."
    >
      <ContentCard title="Image handling">
        <p>The browser sends the selected PNG/JPEG bytes to the configured API only when you submit. The API enforces a request-size bound and closes the upload after validation. The multipart framework may temporarily spool larger request bodies to the operating system&apos;s temporary storage; the upload is closed and that temporary file is not retained by the application.</p>
        <p>Do not upload identifiable or sensitive images unless you are authorized and understand the infrastructure running the service.</p>
      </ContentCard>
      <ContentCard title="Logs and third parties">
        <p>Application code does not log the image, original filename, or patient metadata and does not send image content to a third-party AI provider. Reverse proxies, hosting platforms, or operators may have independent request-logging and retention settings; configure and review them before deployment.</p>
      </ContentCard>
      <ContentCard title="Deletion and local development">
        <p>Because the application creates no persistent image record, there is no app-level upload history to delete. Your browser may keep a local preview until the page is closed or the selected file is replaced. Hosting, operating-system, and backup retention are configured separately.</p>
        <p>This prototype has not been assessed for any regulatory or privacy certification.</p>
      </ContentCard>
    </ContentPage>
  );
}
