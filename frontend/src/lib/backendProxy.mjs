const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";
const DEFAULT_TIMEOUT_MS = 65000;

const PUBLIC_MESSAGES = {
  BACKEND_UNAVAILABLE: "Dịch vụ tư vấn đang tạm thời chưa sẵn sàng.",
  BACKEND_TIMEOUT: "Dịch vụ tư vấn đang mất nhiều thời gian hơn dự kiến.",
  INVALID_BACKEND_RESPONSE: "Dịch vụ tư vấn trả về phản hồi không hợp lệ.",
  RAG_UNAVAILABLE: "Dịch vụ tư vấn đang tạm thời chưa sẵn sàng.",
  LLM_TIMEOUT: "Yêu cầu xử lý mất nhiều thời gian hơn dự kiến.",
  LLM_AUTHENTICATION_FAILED:
    "Dịch vụ xử lý câu hỏi đang tạm thời chưa sẵn sàng.",
  LLM_RATE_LIMITED: "Dịch vụ xử lý câu hỏi đang tạm thời bận.",
  LLM_UNAVAILABLE: "Dịch vụ xử lý câu hỏi đang tạm thời chưa sẵn sàng.",
  LLM_PROVIDER_ERROR: "Dịch vụ xử lý câu hỏi đang tạm thời chưa sẵn sàng.",
  INTERNAL_ERROR: "Không thể xử lý yêu cầu lúc này.",
};

const SAFE_UPSTREAM_CODES = new Set(Object.keys(PUBLIC_MESSAGES));

function positiveInteger(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

export function backendBaseUrl(env = process.env) {
  return (env.BACKEND_URL || DEFAULT_BACKEND_URL).replace(/\/+$/, "");
}

export function backendTimeoutMs(env = process.env) {
  return positiveInteger(env.BACKEND_REQUEST_TIMEOUT_MS, DEFAULT_TIMEOUT_MS);
}

export function structuredError(code, status) {
  return Response.json(
    {
      error: {
        code,
        message: PUBLIC_MESSAGES[code] || PUBLIC_MESSAGES.INTERNAL_ERROR,
      },
    },
    { status },
  );
}

function normalizedUpstreamCode(status, payload) {
  const suppliedCode = payload?.detail?.code || payload?.error?.code;
  if (SAFE_UPSTREAM_CODES.has(suppliedCode)) return suppliedCode;
  if (status === 503) return "RAG_UNAVAILABLE";
  if (status === 504) return "LLM_TIMEOUT";
  return "INTERNAL_ERROR";
}

export async function proxyBackend(
  path,
  {
    method = "GET",
    body,
    fetchImpl = fetch,
    timeoutMs = backendTimeoutMs(),
    baseUrl = backendBaseUrl(),
  } = {},
) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  let upstream;
  try {
    upstream = await fetchImpl(`${baseUrl}${path}`, {
      method,
      headers: {
        accept: "application/json",
        ...(body === undefined ? {} : { "content-type": "application/json" }),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      cache: "no-store",
      signal: controller.signal,
    });
  } catch (error) {
    clearTimeout(timeout);
    const timedOut = controller.signal.aborted || error?.name === "AbortError";
    return structuredError(
      timedOut ? "BACKEND_TIMEOUT" : "BACKEND_UNAVAILABLE",
      timedOut ? 504 : 503,
    );
  }

  const contentType = upstream.headers.get("content-type") || "";
  if (upstream.ok && contentType.includes("text/event-stream")) {
    if (!upstream.body) {
      clearTimeout(timeout);
      return structuredError("INVALID_BACKEND_RESPONSE", 502);
    }

    const reader = upstream.body.getReader();
    const stream = new ReadableStream({
      async pull(streamController) {
        try {
          const { done, value } = await reader.read();
          if (done) {
            clearTimeout(timeout);
            streamController.close();
            return;
          }
          streamController.enqueue(value);
        } catch (error) {
          clearTimeout(timeout);
          streamController.error(error);
        }
      },
      async cancel(reason) {
        clearTimeout(timeout);
        controller.abort();
        await reader.cancel(reason);
      },
    });

    return new Response(stream, {
      status: upstream.status,
      headers: {
        "content-type": contentType,
        "cache-control":
          upstream.headers.get("cache-control") || "no-cache, no-transform",
      },
    });
  }

  let responseText;
  try {
    responseText = await upstream.text();
  } catch (error) {
    const timedOut = controller.signal.aborted || error?.name === "AbortError";
    return structuredError(
      timedOut ? "BACKEND_TIMEOUT" : "BACKEND_UNAVAILABLE",
      timedOut ? 504 : 503,
    );
  } finally {
    clearTimeout(timeout);
  }

  let payload;
  try {
    payload = responseText ? JSON.parse(responseText) : {};
  } catch {
    return structuredError("INVALID_BACKEND_RESPONSE", 502);
  }

  if (!upstream.ok) {
    return structuredError(
      normalizedUpstreamCode(upstream.status, payload),
      upstream.status,
    );
  }

  return Response.json(payload, { status: upstream.status });
}
