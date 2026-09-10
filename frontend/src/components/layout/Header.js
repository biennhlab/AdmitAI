import Link from 'next/link';
import styles from './Header.module.css';

export default function Header() {
  return (
    <header className={styles.header}>
      <div className={`container ${styles.headerContainer}`}>
        <div className={styles.logoContainer}>
          <Link href="/">
            <img 
              src="https://tuyensinh.ptit.edu.vn/wp-content/uploads/sites/4/2024/08/congthongtintuyensinh.png" 
              alt="PTIT - Tuyển sinh" 
              className={styles.logo}
            />
          </Link>
        </div>
        
        <nav className={styles.nav}>
          <ul className={styles.navList}>
            <li className={styles.navItem}><Link href="/">Trang chủ</Link></li>
            <li className={styles.navItem}><Link href="#">Giới thiệu</Link></li>
            <li className={styles.navItem}><Link href="#">Thông báo</Link></li>
            <li className={styles.navItem}><Link href="#">Tin tức</Link></li>
            <li className={styles.navItem}><Link href="#">Đề án tuyển sinh</Link></li>
            <li className={styles.navItem}><Link href="#">Quy đổi điểm</Link></li>
            <li className={styles.navItemBtn}>
              <a href="https://xettuyen.ptit.edu.vn/" className="btn btn-primary" target="_blank" rel="noopener noreferrer">
                Nộp hồ sơ trực tuyến
              </a>
            </li>
          </ul>
        </nav>
        
        {/* Mobile menu toggle (placeholder for now) */}
        <button className={styles.mobileToggle}>
          <i className="fa-solid fa-bars"></i>
        </button>
      </div>
    </header>
  );
}
