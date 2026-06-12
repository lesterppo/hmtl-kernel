use criterion::{black_box, Criterion};
use hmtl_kernel::types::Fp8;
use hmtl_kernel::cmtip::LinearAdapter;

pub fn bench_fp8_conversion(c: &mut Criterion) {
    c.bench_function("fp8_to_f32", |b| {
        let val = Fp8::ONE;
        b.iter(|| black_box(val).to_f32());
    });
    
    c.bench_function("f32_to_fp8", |b| {
        let val = 0.5_f32;
        b.iter(|| Fp8::from_f32(black_box(val)));
    });
}

pub fn bench_linear_adapter(c: &mut Criterion) {
    let adapter = LinearAdapter::new(128, 64);
    let input: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
    
    c.bench_function("linear_adapter_project", |b| {
        b.iter(|| adapter.project(black_box(&input)));
    });
}

criterion::criterion_group!(benches, bench_fp8_conversion, bench_linear_adapter);
criterion::criterion_main!(benches);
