# TMDB API Integration Setup

## Cách lấy TMDB API Key

1. Truy cập [The Movie Database (TMDB)](https://www.themoviedb.org/)
2. Tạo tài khoản miễn phí hoặc đăng nhập
3. Vào **Settings** → **API**
4. Yêu cầu API key (thường được chấp thuận ngay lập tức)
5. Copy API key của bạn

## Cấu hình trong ứng dụng

### 1. Tạo file .env.local

Tạo file `.env.local` trong thư mục gốc của project với nội dung:

```env
# TMDB API Configuration
VITE_TMDB_API_KEY=your_actual_api_key_here

# Backend API Configuration (nếu cần)
VITE_API_BASE_URL=http://localhost:8080/api
```

### 2. Thay thế API key

Thay `your_actual_api_key_here` bằng API key thực tế mà bạn đã lấy từ TMDB.

### 3. Restart ứng dụng

Sau khi thêm file `.env.local`, restart ứng dụng để load environment variables:

```bash
npm run dev
```

## Cách sử dụng

1. Mở trang Admin Management
2. Click **Add Movie**
3. Click nút **Import from TMDB** ở góc phải trên
4. Tìm kiếm phim theo tên hoặc chọn từ danh sách phổ biến
5. Click vào phim muốn import
6. Dữ liệu sẽ được tự động điền vào form
7. Chỉnh sửa thông tin nếu cần và click **Create Movie**

## Tính năng

- **Tìm kiếm phim**: Tìm kiếm theo tên phim
- **Danh sách phổ biến**: Xem phim phổ biến, đánh giá cao, đang chiếu, sắp ra mắt
- **Tự động điền**: Tự động điền thông tin phim, diễn viên, đạo diễn, thể loại
- **Tải hình ảnh**: Tự động tải poster và backdrop từ TMDB
- **Chuyển đổi dữ liệu**: Chuyển đổi dữ liệu TMDB sang format của ứng dụng

## Lưu ý

- API key TMDB miễn phí có giới hạn 1000 requests/ngày
- Hình ảnh được tải về và lưu dưới dạng File object
- Dữ liệu được chuyển đổi để phù hợp với cấu trúc database của ứng dụng
- Có thể chỉnh sửa thông tin sau khi import từ TMDB
