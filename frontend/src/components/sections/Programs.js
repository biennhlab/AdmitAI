import styles from './Programs.module.css';

const categories = [
  {
    id: 1,
    title: 'Công nghệ thông tin',
    description: 'Các chương trình đào tạo về CNTT, Khoa học máy tính, Hệ thống thông tin.',
    icon: 'fa-solid fa-laptop-code'
  },
  {
    id: 2,
    title: 'Điện - Điện tử - Viễn thông',
    description: 'Bao gồm Kỹ thuật điện tử viễn thông, Công nghệ IoT, Robot & Trí tuệ nhân tạo.',
    icon: 'fa-solid fa-microchip'
  },
  {
    id: 3,
    title: 'Kinh tế - Quản trị',
    description: 'Quản trị kinh doanh, Kế toán, Marketing, Thương mại điện tử.',
    icon: 'fa-solid fa-chart-line'
  },
  {
    id: 4,
    title: 'Truyền thông - Báo chí',
    description: 'Công nghệ đa phương tiện, Truyền thông đa phương tiện, Báo chí.',
    icon: 'fa-solid fa-camera-retro'
  }
];

export default function Programs() {
  return (
    <section id="chuong-trinh" className={`section ${styles.programsSection}`}>
      <div className="container">
        <h2 className="section-title">Ngành / Chương trình đào tạo</h2>
        
        <div className={styles.grid}>
          {categories.map((cat) => (
            <div key={cat.id} className={styles.card}>
              <div className={styles.iconWrapper}>
                <i className={cat.icon}></i>
              </div>
              <h3 className={styles.cardTitle}>{cat.title}</h3>
              <p className={styles.cardDesc}>{cat.description}</p>
              <a href="#" className={styles.cardLink}>
                Xem chi tiết <i className="fa-solid fa-arrow-right"></i>
              </a>
            </div>
          ))}
        </div>
        
        <div className={styles.actions}>
          <a href="#" className="btn btn-primary">Xem toàn bộ mã ngành & chỉ tiêu 2026</a>
        </div>
      </div>
    </section>
  );
}
