import styles from './Hero.module.css';

export default function Hero() {
  return (
    <section className={styles.hero}>
      <div className={styles.heroOverlay}></div>
      <div className={`container ${styles.heroContainer}`}>
        <div className={styles.heroContent}>
          <h1 className={`${styles.heroTitle} fade-in`}>
            TUYỂN SINH HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG
          </h1>
          <p className={`${styles.heroSubtitle} fade-in`} style={{ animationDelay: '0.1s' }}>
            Nộp hồ sơ trực tuyến để không bỏ lỡ bất kỳ cơ hội nào trở thành sinh viên của PTIT
          </p>
          <div className={`${styles.heroActions} fade-in`} style={{ animationDelay: '0.2s' }}>
            <a href="https://xettuyen.ptit.edu.vn/" className={`btn btn-primary ${styles.btnLarge}`} target="_blank" rel="noopener noreferrer">
              Đăng ký xét tuyển trực tuyến
            </a>
            <a href="#chuong-trinh" className={`btn ${styles.btnOutline}`}>
              Xem ngành đào tạo
            </a>
          </div>
        </div>
      </div>
      
      {/* Stats Banner */}
      <div className={styles.statsBanner}>
        <div className={`container ${styles.statsGrid}`}>
          <div className={styles.statItem}>
            <i className="fa-solid fa-trophy"></i>
            <div>
              <h4>TOP 1</h4>
              <p>Trường đào tạo CNTT tại Việt Nam</p>
            </div>
          </div>
          <div className={styles.statItem}>
            <i className="fa-solid fa-users"></i>
            <div>
              <h4>25.000+</h4>
              <p>Sinh viên đang theo học</p>
            </div>
          </div>
          <div className={styles.statItem}>
            <i className="fa-solid fa-briefcase"></i>
            <div>
              <h4>98%</h4>
              <p>Sinh viên có việc làm sau khi ra trường</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
