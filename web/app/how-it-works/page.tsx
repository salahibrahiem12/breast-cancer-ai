import type { Metadata } from "next";
import { ContentCard, ContentPage } from "@/components/content-page";

export const metadata: Metadata = { title: "How It Works" };

export default function HowItWorksPage() {
  return (
    <ContentPage
      eyebrow="Plain-language overview"
      title="A cautious pipeline, with clear stop points."
      intro="The software separates image-file checks from medical image interpretation. The first is implemented; the second is deliberately unavailable."
    >
      <ContentCard title="1. Upload validation">
        <p>The API checks the actual PNG/JPEG container, successful decoding, file-size limit, pixel bounds, and a few obvious low-information cases. These checks can catch malformed uploads; they cannot establish that an image shows breast ultrasound.</p>
      </ContentCard>
      <ContentCard title="2. Eligibility gate">
        <p>A separate eligibility model would need diverse breast ultrasound and unrelated-image examples, patient-grouped development data, and an independent evaluation. Those data and measured results are not available, so the gate abstains.</p>
      </ContentCard>
      <ContentCard title="3. Research model pipeline">
        <p>Preparation joins exact and perceptually similar images before splitting and groups by patient when a complete mapping is supplied. Training applies augmentation only to training data; checkpoint selection and temperature fitting use validation data. The test set is evaluated only afterward.</p>
      </ContentCard>
      <ContentCard title="4. Result">
        <p>Today, every decodable upload returns “Unable to assess.” The service does not emit a class, probability, lesion outline, size, BI-RADS category, or treatment recommendation.</p>
      </ContentCard>
    </ContentPage>
  );
}
