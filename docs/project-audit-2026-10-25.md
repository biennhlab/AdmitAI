# Rà soát AdmitAI — 25/10/2026

**Kết luận:** bộ kiểm thử hiện có chạy qua, nhưng project vẫn còn lỗi về độ tin cậy câu trả lời, giới hạn tài nguyên, ingestion và triển khai. Chưa đủ bằng chứng để nghiệm thu toàn bộ sản phẩm hoặc xác nhận sẵn sàng triển khai công khai. Báo cáo gồm **18 phát hiện kỹ thuật** và **4 nhóm chức năng chưa hoàn thành**; không coi các chức năng trên roadmap là lỗi hồi quy.

Rà soát trạng thái working tree hiện tại, gồm cả thay đổi chưa commit. Không sửa mã ứng dụng, không chạy ingestion, không đổi index/collection/database, không khởi động hoặc dừng dịch vụ, không gọi Google AI và không tải model. Các kiểm chứng HTTP dùng ASGI trong tiến trình với dependency giả lập. GitNexus được dùng để định hướng luồng gọi; kết luận đối chiếu trực tiếp với source hiện tại vì graph không phản ánh đầy đủ thay đổi chưa commit.

## Kết quả kiểm tra

| Kiểm tra | Kết quả | Giới hạn ý nghĩa |
|---|---|---|
| Backend `pytest tests` | **674 passed**, khoảng 28 giây | Chạy trên môi trường Python đang cài, có nhiều dependency giả lập; không chứng minh runtime thật hoạt động |
| Frontend Vitest | **29 passed**, 3 file | Fetch và provider được giả lập |
| Frontend production build | **Pass** | Build thành công không xác nhận kết nối backend |
| ESLint | 0 lỗi, **3 cảnh báo** | Font ở layout; hai thẻ `img` |
| `npm audit --json` | **9 package bị báo: 1 critical, 6 high, 2 moderate** | Bao gồm cả dependency phát triển; advisory của package không đồng nghĩa đường khai thác tồn tại trong ứng dụng |
| BM25 golden evaluation | **35 case**, 1 lượt/case | Dense branch rỗng; không gọi LLM; không đo faithfulness câu trả lời |
| Index local | Module 3, **3.185 searchable chunks**, 1.119 parent chunks | Manifest có revision/fingerprint; không có load/chunk/embedding error được ghi nhận |
| HTTP localhost | Các cổng **8000, 6333, 3000 từ chối kết nối** | Trạng thái tại thời điểm audit; không coi việc dịch vụ đang dừng là lỗi mã nguồn |
| Quét mẫu secret trong file Git theo dõi | Không có kết quả với mẫu HF, Google API key, OpenAI key, private key | Không quét toàn bộ Git history, không xác nhận credential cũ đã bị thu hồi |

Lượt chạy đầu bị sandbox chặn tạo tiến trình frontend và tạo thư mục tạm backend. Sau khi chạy ngoài giới hạn đó, build và test đều qua. Các lỗi quyền ban đầu không được tính là lỗi của project. Backend có 17.637 warning, phần lớn deprecation trong dependency; chưa phân loại từng warning.

**Artifact cục bộ:** [BM25 evaluation](../test-results/audit-20261025-bm25.json), [npm audit](../test-results/audit-20261025-npm.json), [mã kiểm chứng](../test-results/audit-20261025-probes.py), [kết quả kiểm chứng](../test-results/audit-20261025-probes.json). `test-results` được Git ignore theo cấu hình project.

## Phát hiện ưu tiên xử lý

P1: nên xử lý trước khi nghiệm thu chatbot hoặc mở dịch vụ ra mạng. P2: ảnh hưởng tính đúng, khả năng vận hành hoặc khả năng tái lập. P3: hoàn thiện tài liệu/UX. Mức ưu tiên của project khác với severity do công cụ advisory gán.

### F01 — P1 — Grounding chấp nhận số liệu gán nhầm chương trình

**Vị trí:** `src/generation/grounding.py:232`, `:237`, `:258`.

Validator kiểm tra tập hợp con số và độ trùng từ trên toàn bộ các nguồn được dẫn. Nó chưa ràng buộc quan hệ giữa chương trình, cơ sở, điều kiện và giá trị trong đúng câu/dòng nguồn.

**Đã tái hiện:** nguồn giả lập ghi “Học phí đại trà 35,9 triệu đồng/năm. Học phí chất lượng cao 58,3 triệu đồng/năm.” Câu trả lời “Học phí đại trà 58,3 triệu đồng/năm [1].” vẫn được `validate_answer` chấp nhận. Con số có thật nhưng thuộc chương trình khác.

Đây là giới hạn có ảnh hưởng trực tiếp tới tư vấn tuyển sinh. README đã thừa nhận kiểm tra chưa phải semantic entailment, nhưng test hiện tại chưa bảo vệ trường hợp đảo quan hệ này. Cần kiểm chứng theo câu/dòng và cặp thực thể–giá trị; thêm case đảo học phí, đảo SAT/ACT, đảo cơ sở vào đánh giá. Không nới ngưỡng overlap để xử lý vấn đề này.

### F02 — P1 — Câu từ chối an toàn do LLM sinh bị biến thành lỗi 502

**Vị trí:** `src/generation/rag_chain.py:808`; `src/generation/grounding.py:228`; `api/routers/chat.py:87`.

Khi retrieval có tài liệu gần chủ đề nhưng không đủ để trả lời, prompt cho phép model nói chưa tìm thấy thông tin. Tuy nhiên, mọi claim đều bị buộc có citation; không có nhánh xử lý câu từ chối hợp lệ sau generation. Phân loại `claim_shape=abstention` chỉ dùng để log.

**Đã tái hiện qua API:** model giả lập trả đúng `FALLBACK_ANSWER`; cả `Accept: application/json` và `Accept: text/event-stream` nhận **502 / GENERATION_UNGROUNDED**. Người dùng thấy thông báo lỗi và được yêu cầu diễn đạt lại dù model đã từ chối đúng.

Cần đưa câu từ chối đã kiểm chứng về response hợp lệ, không citation, route phù hợp. Nhánh đó phải phân biệt câu từ chối thuần túy với câu vừa từ chối vừa chèn khẳng định không có nguồn.

### F03 — P1 — Public API chưa thực thi rate limit và giới hạn công việc

**Vị trí:** `api/main.py:34`; `api/routers/chat.py:287`, `:301`, `:320`, `:322`; `api/routers/auth.py:13`; `api/schemas.py:20`.

Đã tạo SlowAPI Limiter nhưng không cấu hình default limit hay route decorator. Kiểm chứng trạng thái runtime cho thấy **0 default limits, 0 route limits**. `ChatRequest.message` cũng không có `max_length`: chuỗi **1.000.000 ký tự** vẫn hợp lệ ở lớp schema.

Mỗi request SSE tạo một thread riêng, dùng queue không giới hạn; không có cơ chế hủy worker khi client ngắt. JSON đưa công việc vào thread pool nhưng chưa có giới hạn admission theo năng lực model. Login cũng chưa có giới hạn thử mật khẩu. Có thể làm tăng CPU/RAM, chi phí provider và gây cạn tài nguyên khi request đồng thời.

Cần giới hạn body/message, áp dụng rate limit thật cho chat/login, đặt semaphore hoặc hàng đợi có giới hạn cho inference, deadline toàn request và cơ chế dừng/giải phóng khi client hủy. Không chạy flood test trên runtime thật trong audit này.

### F04 — P1 — Ingest mặc định có thể ghi đè index đang được ghim

**Vị trí:** `scripts/ingest.py:252`, `:255`, `:415`; `src/config.py:60`, `:69`.

CLI mặc định chọn **module 1**, còn `--index-dir` mặc định lấy `settings.INDEX_DIR`, đang trỏ vào index Module 3 được hỗ trợ duy nhất. Nếu chạy `python scripts/ingest.py` và embedding thành công, script có thể lưu snapshot Module 1 vào đúng thư mục hiện tại. API sau đó từ chối vì yêu cầu manifest Module 3.

README cảnh báo không rebuild index nhưng CLI chưa thực thi điều kiện bảo vệ này. Module 2 cũng có thể nhắm cùng thư mục nếu không truyền đường dẫn khác. **Không chạy lệnh ghi đè để tái hiện.** Kết luận dựa trên các giá trị mặc định và đường lưu cuối hàm.

Cần đường mặc định riêng theo module và từ chối ghi vào snapshot được bảo vệ. Migration phải có đích mới rõ ràng và quy trình kiểm tra trước khi chuyển cấu hình.

### F05 — P1 — Docker Compose chưa cấp đủ điều kiện cho stack hiện tại

**Vị trí:** `Dockerfile:6`; `docker-compose.yml:11`, `:22`; `src/config.py:16`, `:73`; `api/main.py:404`.

Image không copy `models`, Compose không mount model cache, trong khi cấu hình local bắt buộc `MODELS_LOCAL_ONLY=true`. Với `.env.example` và image này, container không có BGE-M3 snapshot để khởi tạo RAG. Volume `data` không cung cấp thư mục `models` mà code mặc định sử dụng.

Compose cũng chưa có service Next.js. Backend mount `frontend/chatbot`, nhưng thư mục đó chỉ có `app.js`, `style.css`, **không có `index.html`**. Kiểm chứng ASGI `GET /` trả **404**. Vì vậy không thể coi Compose hiện tại là cách chạy toàn bộ giao diện và backend.

Cần provision/mount model và index đúng đường dẫn, đóng gói frontend Next.js thành service hoặc chỉ rõ cách chạy riêng. Docker build/Compose thật chưa được chạy trong audit; không khẳng định thêm lỗi apt/image chưa kiểm chứng.

### F06 — P1 có điều kiện — JWT vẫn có khóa mặc định công khai

**Vị trí:** `src/config.py:23`; `src/auth/security.py:24`; `src/auth/dependencies.py:18`.

`JWT_SECRET_KEY` có giá trị mặc định `secret`; không có validator từ chối khóa mặc định, placeholder hoặc khóa quá yếu. Nếu người triển khai bỏ sót cấu hình, token HS256 có thể được tự ký với khóa công khai này và username tồn tại để đi qua xác thực.

Audit không đọc hoặc công bố giá trị secret hiện tại; **chưa xác nhận môi trường đang dùng khóa yếu**. Đây là đường cấu hình nguy hiểm đã xác nhận trong source. Cần bắt buộc secret đủ mạnh và fail startup khi không hợp lệ. Không dùng attestation của local launcher thay cho validation ở backend.

### F07 — P2 — Dependency có advisory; backend test chưa chạy trên đúng bộ yêu cầu

**Vị trí:** `frontend/package.json:13`, `frontend/package-lock.json`; `requirements.txt:2`, `:11`, `:32`, `:44`.

`npm audit` báo 9 package: `next` critical; `sharp`, `source-map-js`, `brace-expansion`, `undici`, `vite`, `zod` high; `@vitest/mocker`, `vitest` moderate. Một phần nằm trong dev dependencies. Chi tiết version, đường phụ thuộc và advisory nằm trong artifact npm.

Next.js **16.3.4** thuộc dải advisory critical của Node `next/og` ImageResponse. Advisory yêu cầu ứng dụng đưa đầu vào do attacker kiểm soát vào SVG khi sinh ảnh; source hiện tại **không thấy dùng `next/og`/ImageResponse**, nên chưa có bằng chứng project có đường RCE đó. Không diễn giải severity của package thành một vụ khai thác đã xác nhận. [Advisory chính thức của Next.js](https://github.com/vercel/next.js/security/advisories/GHSA-vcvr-r3jv-pc5j).

Backend đang cài `starlette==0.46.2`, nằm trong dải advisory về blocking event loop khi multipart file rollover; bản vá là 0.47.2. FastAPI 0.115.14 đang cài yêu cầu `starlette<0.47.0`, nên cần nâng bộ tương thích thay vì ép riêng Starlette. [Advisory chính thức của Starlette](https://github.com/Kludex/starlette/security/advisories/GHSA-2c2j-9gv5-cj73).

`python-jose==3.3.0` đang cài và requirements khóa ở nhánh 3.3. Bản 3.4.0 đã sửa CVE-2024-33663 và CVE-2024-33664. Đường auth hiện dùng HS256 với allowlist; audit không xác nhận các CVE đó khai thác được qua đường auth hiện tại. [Release notes chính thức](https://github.com/mpdavis/python-jose/releases/tag/3.4.0).

Môi trường test là **Python 3.14.6 / openai 2.15.0 / httpx 0.28.1**, trong khi Docker là Python 3.10, requirements yêu cầu **openai 1.14.* / httpx 0.27.***. Do đó 674 test pass chưa chứng minh bộ dependency được khai báo sẽ pass. Cần lock backend và chạy CI trên đúng môi trường triển khai, sau đó nâng dependency có kiểm soát. Không chạy `npm audit fix` hoặc nâng package trong audit này.

### F08 — P2 — Session giữ RAM không giới hạn và mất ngữ cảnh khi restart

**Vị trí:** `src/generation/session_memory.py:9`, `:14`, `:26`, `:46`; `api/routers/chat.py:280`.

`get_history(max_turns=5)` chỉ cắt bản sao trả về, không cắt dữ liệu lưu. Dictionary session và danh sách message không có TTL, giới hạn số phiên hay giới hạn độ dài lịch sử. Session rỗng được tạo trước generation, kể cả khi request sau đó thất bại.

Mặc dù message được ghi DB, API không nạp lại lịch sử từ DB. Sau restart, session ID đang lưu ở trình duyệt không còn trong dictionary nên API tạo ID mới và mất ngữ cảnh. Hiện tượng tương tự xảy ra giữa nhiều process; profile một worker chỉ tránh một phần trường hợp đó.

Cần giới hạn/thu hồi session và quy định rõ nguồn lịch sử bền vững. Nếu DB là nguồn lịch sử, cần đọc lại khi session chưa có trong cache.

### F09 — P2 — Retrieval không dùng lịch sử hội thoại

**Vị trí:** `src/generation/rag_chain.py:201`, `:212`, `:267`.

`session_history` chỉ được thêm vào prompt sau khi tìm và chọn tài liệu xong. Query tìm kiếm, lọc scope và lexical evidence chỉ dựa vào câu hỏi mới. QueryRewriter hiện tại cũng không nhận lịch sử; chỉ bật cờ rewrite sẽ không giải quyết được vấn đề.

**Đã tái hiện:** cùng câu “Còn cơ sở phía Nam?” với hai lịch sử khác nhau, một về học phí và một về điểm chuẩn CNTT, tạo hai truy vấn tìm kiếm giống hệt nhau. Chưa đo tỷ lệ trả lời sai thực tế với provider; vấn đề đã xác nhận là mất thông tin chủ đề ở bước retrieval.

Cần giải quyết tham chiếu và tạo câu tìm kiếm độc lập dựa trên các lượt gần nhất, giữ nguyên mã ngành/năm/cơ sở. Thêm đánh giá nhiều lượt thay vì chỉ kiểm tra history được truyền tới mock RAG.

### F10 — P2 — Startup chấp nhận Qdrant sai nội dung nếu số lượng vẫn khớp

**Vị trí:** `api/main.py:138`, `:151`; `scripts/local_runtime.py:118`; đối chiếu `scripts/preflight_index_migration.py:75`.

Backend startup kiểm tra collection, count, dimension và distance nhưng không so point ID/payload với snapshot BM25. Launcher kiểm tra status/count; chưa gọi đối chiếu đầy đủ mà script migration preflight đã có. Collection cùng tên, cùng số lượng nhưng chứa dữ liệu khác có thể được coi là sẵn sàng và làm hai nhánh dense/sparse lệch nhau.

**Đã tái hiện bằng client giả lập:** count khớp, point ID/payload khác; `load_hybrid_retriever` vẫn trả retriever thành công, không gọi scroll để đối chiếu. Không xác nhận collection thật đang có sai lệch vì Qdrant đang dừng.

Cần đối chiếu ID/payload hoặc xác minh một digest corpus/build đáng tin cậy trong quy trình start/cutover; tái sử dụng logic preflight hiện có. Chỉ tăng số lần kiểm tra count không phát hiện được trường hợp này.

### F11 — P2 — Thay nội dung Markdown có thể không đổi corpus fingerprint

**Vị trí:** `src/ingestion/loader.py:133`; `scripts/ingest.py:37`, `:68`.

Loader dùng `setdefault` cho `content_hash`, nên giữ nguyên hash đã có trong frontmatter. `corpus_fingerprint` hash metadata và doc ID, không tự hash lại `document.content`. Nếu sửa nội dung nhưng để hash frontmatter cũ, snapshot có thể bị nhận là không thay đổi và tiếp tục trả dữ liệu cũ.

**Đã tái hiện không ghi file corpus:** hai bản Markdown cùng metadata/hash cũ nhưng nội dung học phí đổi từ 35.9 thành 99.9 tạo fingerprint giống nhau. Trường hợp này cần hash frontmatter bị cũ; crawler cập nhật hash đúng sẽ không gặp nó.

Cần tính/kiểm tra content hash từ body thực tế khi load và đưa nội dung thực vào fingerprint, kèm kế hoạch tương thích với index/golden được ghim. Không sửa fingerprint của index đang dùng bằng tay.

### F12 — P2 — Helper kiểm chứng API vẫn đọc JSON trong khi API mặc định trả SSE

**Vị trí:** `scripts/verify_rag.py:66`, `:67`, `:86`, `:99`; `frontend/chatbot/app.js:181`; `api/routers/chat.py:108`.

`verify_rag.py --api-url ...` không gửi `Accept: application/json`, nhưng gọi `response.json()`. HTTPX mặc định chấp nhận wildcard; backend giữ SSE khi wildcard/không có preference. Với câu trả lời thành công, helper có thể lỗi JSONDecodeError thay vì tạo báo cáo. Legacy chatbot cũng có sự lệch hợp đồng tương tự.

Cần gửi Accept JSON rõ ràng hoặc parse SSE. Các lần gọi follow-up và negative trong helper cũng cần cùng hợp đồng. Frontend Next.js hiện tại đã parse SSE; phát hiện này không áp dụng cho chatbot Next.js.

### F13 — P2 — Độ phủ bằng chứng golden của nhánh BM25 còn thấp

**Vị trí:** `scripts/evaluate_domain_lite.py`; `eval/golden_dataset.json`; artifact `test-results/audit-20261025-bm25.json`.

Kết quả thực chạy trên corpus hiện tại:

| Metric | Lexical baseline | Domain lite |
|---|---:|---:|
| Recall exact chunk ID trung bình | 4,84% | 12,90% |
| Độ phủ bằng chứng tính cả parent tương đương | 53,23% | **55,48%** |
| MRR có parent tương đương | 0,3231 | 0,4091 |
| Negative false-positive, 4 negative case | 100% | **0%** |
| Case có bằng chứng chuẩn nhưng coverage = 0 | — | **13 / 31** |

`admissions_all_methods_2026` chỉ đạt coverage 0,2; `admissions_vact_spt_thresholds` bị route thành `out_of_scope`. Cần điều tra các nhóm assessment thresholds, học bổng, danh sách phương thức và một số ngành/cơ sở.

**Không diễn giải 55,48% thành độ chính xác chatbot.** Dense branch rỗng trong lượt chạy này; nhiều chunk khác có thể chứa dữ kiện tương đương ngoài evidence ID chuẩn. Metric chưa tự kiểm chứng tính tương đương ngữ nghĩa. Cần xem từng case và chạy thêm dense + BM25 thật, sau đó chấm câu trả lời/citation. Không thay golden ID để làm đẹp metric.

### F14 — P2 — Nội dung phương thức tuyển sinh trên trang lệch golden/corpus

**Vị trí:** `frontend/src/components/sections/AdmissionMethods.js:3`; case `admissions_all_methods_2026` trong `eval/golden_dataset.json`.

UI hardcode năm mục gồm tuyển thẳng, THPT, kết hợp, đánh giá năng lực và đánh giá tư duy. Golden/corpus hiện tại mô tả một mục SAT/ACT và gộp đánh giá năng lực/tư duy thành một phương thức. UI không có mục SAT/ACT, nên cùng một project có hai cách trình bày không khớp nhau.

Đây là đối chiếu với dữ liệu local, chưa phải lần xác minh độc lập toàn bộ quy chế tuyển sinh đang công bố. Cần đồng bộ nội dung trang với nguồn chính thức được quản lý, có năm và ngày kiểm tra. Các con số marketing hardcode ở Hero cũng chưa có nguồn trong component; chưa kết luận chúng sai.

### F15 — P2 — Mobile menu và nhiều liên kết điều hướng chưa hoạt động

**Vị trí:** `frontend/src/components/layout/Header.js:35`; `Header.module.css:56`; `frontend/src/components/sections/Programs.js`; `frontend/src/components/layout/Footer.js`.

Ở màn hình <= 1024px, CSS ẩn navigation; nút hamburger chỉ có icon, không có handler/state mở menu. Người dùng mobile không truy cập được menu qua nút này. Nhiều liên kết giới thiệu/thông báo/tin tức/chi tiết ngành/social còn là `#`.

Cần hoàn thiện menu có trạng thái và hỗ trợ bàn phím, rồi thay link placeholder bằng đích thật hoặc bỏ phần chưa có nội dung. Build pass không bắt được lỗi hành vi này.

### F16 — P2 có điều kiện — Compose công bố Qdrant trên mọi interface mà chưa có auth

**Vị trí:** `docker-compose.yml:4`; `api/main.py:141`; `scripts/ingest.py:181`.

Port mapping `6333:6333`, `6334:6334` mặc định bind trên các interface host; Compose không thiết lập API key và client chỉ có host/port. Khi host cho phép truy cập từ mạng, dịch vụ lưu trữ có thể được truy cập trực tiếp, độc lập với JWT của API. Không thử truy cập từ mạng ngoài trong audit.

Với local development, bind rõ `127.0.0.1` hoặc giữ Qdrant trong network nội bộ. Với triển khai từ xa, cần auth, TLS và cấu hình client tương ứng. Không coi firewall chưa kiểm tra là lớp bảo vệ đã có.

### F17 — P2 — Chưa có kiểm tra tích hợp tự động cho môi trường triển khai

**Vị trí:** chưa có `.github/workflows`; `requirements.txt`; `frontend/tests`; `tests/test_api_resilience.py:87`, `:100`; `src/eval/__init__.py`.

Bộ test có giá trị cho contract và các tình huống lỗi nhưng phần lớn boundary API/model/provider dùng mock. Chưa có CI được version control để chứng minh clean install, đúng version Python/dependency, build frontend, Qdrant thật và smoke JSON/SSE. Bộ 674 test không bao gồm một lần tư vấn thật thành công chỉ vì tất cả đang xanh.

Cần CI tối thiểu cho clean install/test/build và một luồng integration có kiểm soát với Qdrant. Provider smoke nên là job riêng theo cấu hình credential, có ngân sách và dữ liệu kiểm tra cố định; không nhúng secret vào artifact. Trong audit này chưa chạy clean install hay provider smoke.

### F18 — P3 — Tài liệu setup và nghiệm thu chưa thống nhất với trạng thái sản phẩm

**Vị trí:** `docs/00-project-overview.md`; `docs/11-deployment-devops.md:50`; `docs/13-acceptance-criteria.md`; `README.md:23`; `docs/index-migration-runbook.md`.

Tài liệu cũ còn mô tả frontend HTML, Claude/RAGAS, mở `frontend/chatbot/index.html` không tồn tại và full stack Docker; README hiện tại mô tả Next.js/Google AI/local launcher. Các yêu cầu và acceptance criteria chưa đánh dấu rõ phần đã làm, chưa làm và phần thay thế bởi local lite.

README yêu cầu index/model đã provision, trong khi hai asset không nằm trong Git; runbook hiện tập trung kiểm tra tính toàn vẹn, chưa đưa người mới tới một nguồn cấp phát có thể tái lập. Cần phân biệt setup từ clean clone với vận hành máy đã được provision, và mô tả cách nhận/khôi phục asset đã được xác minh.

## Chức năng còn thiếu so với tài liệu yêu cầu

Các mục dưới đây là khoảng trống scope đã xác nhận, không bị gộp thành lỗi của thay đổi chưa commit:

| Nhóm | Trạng thái thực tế | Tác động |
|---|---|---|
| Quản trị | Cả 7 handler ở `api/routers/admin.py:10` còn ghi `Phase 0 Stub`; chưa có trang login/dashboard trong frontend | History/documents trả mảng rỗng; upload trả 202 giả; get document trả `indexed` với mọi ID; delete trả thành công nhưng không xóa |
| Escalation, agentic, calculator | Có model Escalation nhưng không có luồng tạo/xử lý thực; `src/routing` chưa có router; tool calling còn `NotImplementedError` | Chưa đáp ứng tính điểm, phối hợp tool, fallback chuyển nhân viên và human handoff |
| Feedback | Nút thumbs-up/down Next.js chưa có `onClick`; API luôn trả `ok`; schema DB chưa có bảng/cột feedback và response chưa cung cấp message ID để gửi | Đã kiểm chứng message ID không tồn tại vẫn được phản hồi 200; chưa có phản hồi thực để đánh giá chất lượng |
| Evaluation đầy đủ | Có 35 golden case và evaluator retrieval; chưa có RAGAS runner thực trong `src/eval` và requirements chưa có RAGAS | Chưa có chuỗi số đo context precision/recall, faithfulness, answer relevancy qua các module theo tiêu chí nghiệm thu |

## Thứ tự khắc phục đề xuất

1. Sửa F01–F02 để câu trả lời sai quan hệ không lọt và câu từ chối an toàn được trả đúng. Bổ sung kiểm chứng bằng dữ liệu tuyển sinh, cả JSON/SSE.
2. Chặn ghi đè index mặc định (F04), đối chiếu index/collection (F10), sửa content fingerprint với kế hoạch migration rõ ràng (F11).
3. Đặt giới hạn request/inference/session, bắt buộc JWT secret hợp lệ và bảo vệ Qdrant trước khi mở mạng (F03, F06, F08, F16).
4. Đồng bộ dependency, xử lý advisory theo đường sử dụng thực tế, thiết lập CI và làm Docker/provisioning có thể tái lập (F05, F07, F17, F18).
5. Cải thiện retrieval nhiều lượt và đánh giá dense + BM25 + generation thật; sửa nội dung/điều hướng frontend (F09, F12–F15).
6. Chốt phạm vi phiên bản nghiệm thu rồi hoàn thiện admin, escalation, feedback, calculator và evaluation tương ứng. Không báo các endpoint stub là chức năng đã hoạt động.

Các mục chưa được kiểm tra trực tiếp: runtime Google AI, chất lượng hybrid dense thật, tải đồng thời, UI trong trình duyệt thật, Docker build/run, clean install và Git-history secret scan. Những mục này cần kiểm chứng tiếp để nghiệm thu; báo cáo không giả định chúng đã đạt hoặc đã thất bại.
