export interface NavItem {
  label: string;
  href: string;
}

export interface Feature {
  id: string;
  title: string;
  description: string;
  image: string;
  tag: string;
}

export enum ScrollDirection {
  UP = 'UP',
  DOWN = 'DOWN'
}