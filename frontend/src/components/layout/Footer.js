import Link from 'next/link';
import styles from './Footer.module.css';

export default function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={`container ${styles.footerContainer}`}>
        <div className={styles.column}>
          <img 
            src="https://tuyensinh.ptit.edu.vn/wp-content/uploads/sites/4/2024/08/Group-792.png" 
            alt="PTIT Logo" 
            className={styles.logo}
          />
          <h3 className={styles.title}>HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG</h3>
          <p><strong>Cơ sở đào tạo Phía Bắc:</strong></p>
          <p>Km10, Đường Nguyễn Trãi, Q.Hà Đông, Hà Nội</p>
          <p><strong>Cơ sở đào tạo Phía Nam:</strong></p>
          <p>11 Nguyễn Đình Chiểu, P. Đa Kao, Q.1 TP Hồ Chí Minh</p>
        </div>
        
        <div className={styles.column}>
          <h3 className={styles.title}>LIÊN HỆ</h3>
          <p><i className="fa-solid fa-phone"></i> 1900 8138 (Phía Bắc)</p>
          <p><i className="fa-solid fa-phone"></i> 1900 7110 (Phía Nam)</p>
          <p><i className="fa-solid fa-envelope"></i> tuyensinh@ptit.edu.vn (Phía Bắc)</p>
          <p><i className="fa-solid fa-envelope"></i> tuyensinh@ptithcm.edu.vn (Phía Nam)</p>
        </div>
        
        <div className={styles.column}>
          <h3 className={styles.title}>LIÊN KẾT NHANH</h3>
          <ul className={styles.links}>
            <li><Link href="/">Trang chủ</Link></li>
            <li><Link href="#">Giới thiệu</Link></li>
            <li><Link href="#">Thông báo</Link></li>
            <li><Link href="#">Đề án tuyển sinh</Link></li>
            <li><Link href="#">Tin tức</Link></li>
          </ul>
        </div>
        
        <div className={styles.column}>
          <h3 className={styles.title}>KẾT NỐI VỚI CHÚNG TÔI</h3>
          <div className={styles.socials}>
            <a href="#" className={styles.socialLink}><i className="fa-brands fa-facebook-f"></i></a>
            <a href="#" className={styles.socialLink}><i className="fa-brands fa-youtube"></i></a>
            <a href="#" className={styles.socialLink}><i className="fa-brands fa-tiktok"></i></a>
          </div>
        </div>
      </div>
      <div className={styles.bottomBar}>
        <div className="container">
          <p>&copy; {new Date().getFullYear()} Học viện Công nghệ Bưu chính Viễn thông. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
}
