import './globals.css'
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import FloatingChatbot from '@/components/chatbot/FloatingChatbot';

export const metadata = {
  title: 'PTIT - Tuyển sinh',
  description: 'Cổng thông tin tuyển sinh Học viện Công nghệ Bưu chính Viễn thông',
}

export default function RootLayout({ children }) {
  return (
    <html lang="vi">
      <head>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" />
      </head>
      <body>
        <Header />
        <main>{children}</main>
        <Footer />
        <FloatingChatbot />
      </body>
    </html>
  )
}
