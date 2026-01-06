import { z } from 'zod';

const envSchema = z.object({
  TURBOPUFFER_API_KEY: z.string(),
  OPENAI_API_KEY: z.string(),
  JINA_API_KEY: z.string(),
});

export const env = envSchema.parse(process.env);
