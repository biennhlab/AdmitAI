import styles from './AdmissionMethods.module.css';

const methods = [
  {
    id: 1,
    title: 'Tuyển thẳng',
    description: 'Theo quy chế tuyển sinh của Bộ GD&ĐT và của Học viện.',
    icon: 'fa-solid fa-award'
  },
  {
    id: 2,
    title: 'Dựa vào điểm thi THPT',
    description: 'Dựa vào kết quả điểm thi THPT năm 2026.',
    icon: 'fa-solid fa-pen-nib'
  },
  {
    id: 3,
    title: 'Kết hợp',
    description: 'Xét tuyển kết hợp giữa chứng chỉ quốc tế và kết quả học tập THPT.',
    icon: 'fa-solid fa-puzzle-piece'
  },
  {
    id: 4,
    title: 'Đánh giá năng lực',
    description: 'Dựa vào kết quả thi ĐGNL của Đại học Quốc gia Hà Nội, ĐHQG TP.HCM.',
    icon: 'fa-solid fa-brain'
  },
  {
    id: 5,
    title: 'Đánh giá tư duy',
    description: 'Dựa vào kết quả thi ĐGTD của Đại học Bách khoa Hà Nội.',
    icon: 'fa-solid fa-lightbulb'
  }
];

export default function AdmissionMethods() {
  return (
    <section className={`section ${styles.methodsSection}`}>
      <div className="container">
        <h2 className="section-title">Phương thức tuyển sinh 2026</h2>
        
        <div className={styles.grid}>
          {methods.map((method) => (
            <div key={method.id} className={styles.card}>
              <div className={styles.iconWrapper}>
                <i className={method.icon}></i>
              </div>
              <div>
                <h3 className={styles.cardTitle}>{method.title}</h3>
                <p className={styles.cardDesc}>{method.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
